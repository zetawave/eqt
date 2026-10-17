package org.equity.app

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.PowerManager
import android.os.SystemClock
import android.system.Os
import android.system.OsConstants
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors

data class Settings(
    val systemPrompt: String = "You are Equity, a helpful assistant running privately on this phone. Answer in the user's language, clearly and concisely.",
    val maxTokens: Int = 512,
    val temperature: Float = 0f,
    // The six big cores (one X4, five A720) on the Exynos 2400: 26% faster decoding than four, measured.
    val threads: Int = 6,
    val context: Int = 4096,
    /** 0 selects the expert-cache budget from available RAM at load time. */
    val expertCacheMiB: Int = 0,
    /** Prompts and images are read on the GPU (Vulkan); answers are still generated on the CPU. */
    val gpu: Boolean = false,
)

object SettingsStore {
    private val flow = MutableStateFlow(Settings())
    val state: StateFlow<Settings> = flow.asStateFlow()

    fun init(context: Context) {
        val p = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val d = Settings()
        flow.value = Settings(
            systemPrompt = p.getString("system", d.systemPrompt) ?: d.systemPrompt,
            maxTokens = p.getInt("max_tokens", d.maxTokens),
            temperature = p.getFloat("temperature", d.temperature),
            threads = p.getInt("threads", d.threads),
            context = p.getInt("context", d.context),
            expertCacheMiB = p.getInt("expert_cache", d.expertCacheMiB),
            gpu = p.getBoolean("gpu", d.gpu),
        )
    }

    fun update(context: Context, change: (Settings) -> Settings) {
        val next = change(flow.value)
        flow.value = next
        context.getSharedPreferences("settings", Context.MODE_PRIVATE).edit()
            .putString("system", next.systemPrompt).putInt("max_tokens", next.maxTokens)
            .putFloat("temperature", next.temperature).putInt("threads", next.threads)
            .putInt("context", next.context).putInt("expert_cache", next.expertCacheMiB)
            .putBoolean("gpu", next.gpu).apply()
    }
}

enum class Phase { Empty, Loading, Ready, Generating, Unloading }

/** The answer being written: its chat, its message and how many tokens have arrived. */
data class Generation(val chatId: String, val answerId: Long, val startedAt: Long, val tokens: Int = 0)

data class EngineState(
    val phase: Phase = Phase.Empty,
    val modelId: String? = null,
    val modelName: String? = null,
    val status: String = "No model loaded",
    val loadPercent: Int? = null,
    val configuration: String? = null,
    /** The loaded model was given its vision projector and accepts images. */
    val vision: Boolean = false,
    /** The catalog model loaded most recently, offered again after an unload. */
    val lastModelId: String? = null,
    val generation: Generation? = null,
    val lastResult: String? = null,
    val lastSpeed: Double? = null,
    val error: String? = null,
) {
    val busy get() = phase == Phase.Loading || phase == Phase.Generating
}

/** Owns the single native engine. Native calls run on one worker thread; only cancellation crosses threads. */
object EngineController {
    private val worker = Executors.newSingleThreadExecutor()
    private val flow = MutableStateFlow(EngineState())
    val state: StateFlow<EngineState> = flow.asStateFlow()
    private var descriptor: ParcelFileDescriptor? = null
    private var modelHash: String? = null
    @Volatile private var cancelled = false

    // Images shown to the model per request: older images in a chat are described in text instead, so a long
    // conversation does not re-encode every photo on each turn.
    private const val MAX_CONTEXT_IMAGES = 4
    // Vision encoder working memory on top of the projector weights while images are encoded (its compute buffer
    // measured 35 MiB at 1024x1024 on the Galaxy).
    private const val VISION_WORK_MIB = 128
    // Free memory left to the rest of the phone once a streamed model is loaded. The admission sum below matched
    // the measured peak RSS of a decode (3,088 MiB predicted 3,126 with a 512 MiB cache), so no further margin
    // is needed; Android also reclaims cached apps and swaps before it would stop the foreground app.
    private const val SYSTEM_RESERVE_MIB = 384
    // GPU mode on the Galaxy (929-token prompt, batch 2048): compute buffers 140 MiB above batch 512, and up to
    // 310 MiB of GPU staging while a prompt is read, released before the answer is written. Measured peak RSS.
    private const val GPU_WORK_MIB = 192
    private const val GPU_STAGING_MIB = 384
    // Answer text reaches the screen at most this often: every update re-lays out the Markdown on the UI thread,
    // which then takes a big core from the compute threads (about two thirds of a core at 60 ms, measured).
    private const val UPDATE_INTERVAL_MS = 200L

    private fun update(change: (EngineState) -> EngineState) = flow.update(change)

    private fun task(action: () -> Unit) {
        cancelled = false
        NativeBridge.prepare()
        worker.execute {
            try {
                action()
            } catch (error: Exception) {
                update { it.copy(error = error.message ?: error.javaClass.simpleName) }
            }
        }
    }

    fun clearError() = update { it.copy(error = null) }

    /** Loads a downloaded catalog model with options tuned for it. */
    fun load(context: Context, model: CatalogModel) {
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        startLoad(app, model.id, model.name)
        task {
            guardLoad {
                val path = ModelFiles.final(app, model.artifact)
                require(ModelFiles.installed(app, model.artifact)) { "${model.name} is not downloaded" }
                val projector = model.projector?.takeIf { ModelFiles.installed(app, it) }
                val visionMiB = projector?.let { (it.sizeBytes shr 20).toInt() + VISION_WORK_MIB } ?: 0
                // Image support keeps room for the projector while it encodes; without that room the model still
                // loads, text-only.
                val withVision = projector?.let {
                    runCatching {
                        loadOptions(app, model.streamed, model.residentMiB, model.contextMiBPer1k, path.length(), settings, visionMiB)
                    }.getOrNull()?.put("mmproj", ModelFiles.final(app, it).absolutePath)
                }
                val options = withVision
                    ?: loadOptions(app, model.streamed, model.residentMiB, model.contextMiBPer1k, path.length(), settings, 0)
                val note = if (projector != null && withVision == null) "images off: not enough free memory" else null
                loadNative(path.absolutePath, options, model.sha256, note)
            }
        }
    }

    /** Loads a user-selected local GGUF through a seekable descriptor, without copying it. */
    fun load(context: Context, uri: Uri) {
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        startLoad(app, null, "Local model")
        task {
            guardLoad {
                val pfd = requireNotNull(app.contentResolver.openFileDescriptor(uri, "r")) { "Cannot open the file" }
                descriptor = pfd
                val stat = Os.fstat(pfd.fileDescriptor)
                require(OsConstants.S_ISREG(stat.st_mode) && stat.st_size > 0) {
                    "Choose a local, seekable file. Cloud providers and pipes are unsupported."
                }
                val magic = ByteArray(4)
                require(Os.pread(pfd.fileDescriptor, magic, 0, 4, 0) == 4 && magic.decodeToString() == "GGUF") {
                    "The selected file is not a GGUF model"
                }
                val sizeMiB = (stat.st_size shr 20).toInt()
                val streamed = sizeMiB + 1536 > availableMiB(app)
                // Small files are hashed for provenance; large streamed artifacts are not re-read on every load.
                var hash: String? = null
                if (!streamed) {
                    update { it.copy(status = "Checking file integrity…") }
                    val digest = MessageDigest.getInstance("SHA-256")
                    val buffer = ByteArray(1 shl 20)
                    var offset = 0L
                    while (offset < stat.st_size) {
                        check(!cancelled) { "Loading cancelled" }
                        val count = Os.pread(pfd.fileDescriptor, buffer, 0, minOf(buffer.size.toLong(), stat.st_size - offset).toInt(), offset)
                        require(count > 0) { "Incomplete model read" }
                        digest.update(buffer, 0, count)
                        offset += count
                    }
                    hash = digest.digest().joinToString("") { "%02x".format(it) }
                }
                // Unknown local files use the larger per-token context cost of the catalog models.
                val options = loadOptions(app, streamed, if (streamed) 2448 else sizeMiB, 112, stat.st_size, settings, 0)
                loadNative("/proc/self/fd/${pfd.fd}", options, hash, null)
            }
        }
    }

    private fun startLoad(context: Context, id: String?, name: String) {
        update {
            it.copy(phase = Phase.Loading, modelId = id, modelName = name, loadPercent = null, error = null,
                status = "Preparing $name…", configuration = null, vision = false, lastModelId = id ?: it.lastModelId)
        }
        InferenceService.start(context)
    }

    /** Any load failure returns the engine to an empty, consistent state. */
    private fun guardLoad(action: () -> Unit) {
        try {
            action()
        } catch (error: Exception) {
            runCatching { NativeBridge.unload() }
            closeDescriptor()
            update {
                EngineState(status = "No model loaded", lastModelId = it.lastModelId,
                    error = if (cancelled) null else (error.message ?: error.javaClass.simpleName))
            }
        }
    }

    private fun availableMiB(context: Context): Int {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        return (memory.availMem shr 20).toInt()
    }

    /** [visionMiB] is memory kept free for image encoding; 0 for a text-only load. */
    private fun loadOptions(context: Context, streamed: Boolean, residentMiB: Int, contextMiBPer1k: Int, fileBytes: Long,
                            settings: Settings, visionMiB: Int): JSONObject {
        val available = availableMiB(context)
        // Context state and compute buffers are allocated at load; admitting without them invites the low-memory killer.
        // Prompt batches of 512 tokens take the tiled GEMM path and need larger compute buffers.
        // GPU mode reads prompts in batches of 2048 tokens (fewer passes over the streamed experts, fuller GPU
        // tiles): larger compute buffers, plus staging for one GPU operation that is released after each prompt.
        val gpuWorkMiB = if (settings.gpu) GPU_WORK_MIB else 0
        val gpuStagingMiB = if (settings.gpu) GPU_STAGING_MIB else 0
        val workMiB = contextMiBPer1k * settings.context / 1024 + 512 + gpuWorkMiB
        val options = JSONObject().put("context", settings.context).put("batch", if (settings.gpu) 2048 else 512)
            .put("threads", settings.threads)
        if (settings.gpu) options.put("gpu", true)
        if (!streamed) {
            val fileMiB = (fileBytes shr 20).toInt()
            require(fileMiB + workMiB + 512 + maxOf(visionMiB, gpuStagingMiB) <= available) {
                "This model needs about ${fileMiB + workMiB + 512 + maxOf(visionMiB, gpuStagingMiB)} MiB of free memory and $available MiB are available. " +
                    "Close some apps or choose a smaller context in Settings."
            }
            return options.put("mmap", true).put("memory_budget_mib", maxOf(2048, fileMiB + 1024) + gpuWorkMiB + gpuStagingMiB)
        }
        // Expert cache from free RAM after shared weights, compute buffers, the transient next-layer allowance used
        // while reading prompts, and system headroom.
        val prefillMiB = 512
        // Images are encoded before the prompt is read and the projector is released before it, so image memory
        // and the prompt-time prefetch allowance are never needed at the same time.
        val transientMiB = maxOf(prefillMiB + gpuStagingMiB, visionMiB)
        val cache = settings.expertCacheMiB.takeIf { it > 0 }
            ?: (available - residentMiB - workMiB - transientMiB - SYSTEM_RESERVE_MIB).coerceIn(512, 3072)
        require(residentMiB + cache + transientMiB + workMiB + SYSTEM_RESERVE_MIB <= available) {
            "This model needs about ${residentMiB + cache + transientMiB + workMiB + SYSTEM_RESERVE_MIB} MiB of free memory and " +
                "$available MiB are available. Close some apps and try again."
        }
        // Measured defaults (docs/decisions.md D018, D020): coalesced expert reads, next-layer prefetch for prompts;
        // the engine keeps experts in reusable slots by default.
        return options.put("mmap", false).put("expert_streaming", true).put("expert_cache_mib", cache)
            .put("io_threads", 16).put("direct_io", true).put("prefetch", true)
            .put("coalesce_kib", 2048).put("coalesce_gap", 1).put("prefill_prefetch_mib", prefillMiB)
            .put("memory_budget_mib", residentMiB + cache + prefillMiB + 1536 + gpuWorkMiB + gpuStagingMiB)
    }

    private fun loadNative(path: String, options: JSONObject, hash: String?, note: String?) {
        NativeBridge.unload()
        val progress = ByteSink { bytes ->
            val percent = bytes.decodeToString().toIntOrNull()
            update { it.copy(loadPercent = percent, status = "Loading ${percent ?: 0}%") }
        }
        var gpuNote: String? = null
        val info = try {
            JSONObject(NativeBridge.load(path.toByteArray(), options.toString().toByteArray(), progress).decodeToString())
        } catch (error: IllegalStateException) {
            when {
                // Storage without O_DIRECT support falls back to buffered reads, and says so.
                options.optBoolean("direct_io") && error.message?.contains("O_DIRECT") == true -> options.put("direct_io", false)
                // A phone without a usable Vulkan GPU, or a build without the GPU backend, runs on the CPU.
                options.optBoolean("gpu") && error.message?.let { "GPU" in it || "Vulkan" in it } == true -> {
                    options.remove("gpu")
                    options.put("batch", 512)
                    gpuNote = "GPU unavailable"
                }
                else -> throw error
            }
            JSONObject(NativeBridge.load(path.toByteArray(), options.toString().toByteArray(), progress).decodeToString())
        }
        modelHash = hash
        val vision = !info.isNull("vision")
        val streaming = info.optJSONObject("expert_streaming")
        val configuration = buildString {
            append("CPU ").append(info.optString("cpu_variant").removePrefix("android_"))
            info.optJSONObject("gpu")?.let { append(" + GPU ").append(it.optString("device")) }
            append(" · ${info.getInt("threads")} threads · ${info.getInt("context")} context")
            if (streaming != null) {
                append(" · ${streaming.getLong("effective_budget_bytes") shr 20} MiB expert cache")
                if (!options.optBoolean("direct_io")) append(" · buffered reads")
            }
            if (vision) append(" · images")
            note?.let { append(" · ").append(it) }
            gpuNote?.let { append(" · ").append(it) }
        }
        update {
            it.copy(phase = Phase.Ready, loadPercent = null, lastResult = null, lastSpeed = null,
                configuration = configuration, status = "Ready", vision = vision)
        }
    }

    private fun closeDescriptor() {
        descriptor?.close()
        descriptor = null
    }

    fun unload() {
        if (flow.value.phase == Phase.Empty) return
        stop()
        update { it.copy(phase = Phase.Unloading, status = "Unloading…") }
        task {
            NativeBridge.unload()
            closeDescriptor()
            update { EngineState(status = "Model unloaded", lastModelId = it.lastModelId) }
        }
    }

    /** Cancels loading or generation; safe from any thread. */
    fun stop() {
        cancelled = true
        NativeBridge.cancel()
    }

    /** Releases the model when Android asks for memory; [message] is shown when the user returns. */
    fun memoryPressure(message: String) {
        if (flow.value.phase == Phase.Empty) return
        stop()
        task {
            NativeBridge.unload()
            closeDescriptor()
            update { EngineState(status = "No model loaded", lastModelId = it.lastModelId, error = message) }
        }
    }

    /**
     * Sends a message in [chatId] (a new chat when null). The answer keeps being written while Equity is in the
     * background: a foreground service holds the process and a wake lock until it is complete.
     */
    fun send(context: Context, chatId: String?, prompt: String, attachments: List<Attachment>): Boolean {
        val text = prompt.trim()
        val engine = flow.value
        if (engine.phase != Phase.Ready || (text.isEmpty() && attachments.isEmpty())) return false
        if (attachments.any { it.kind == AttachmentKind.Image } && !engine.vision) return false
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        val (chat, answerId) = ChatStore.startTurn(chatId, text, attachments)
        update {
            it.copy(phase = Phase.Generating, status = "Reading your message…", error = null,
                generation = Generation(chat.id, answerId, SystemClock.elapsedRealtime()))
        }
        InferenceService.start(app)
        task {
            val messages = JSONArray().put(JSONObject().put("role", "system").put("content", settings.systemPrompt))
            requestMessages(chat, engine.vision).forEach { messages.put(it) }
            val request = JSONObject().put("messages", messages).put("max_tokens", settings.maxTokens)
                .put("temperature", settings.temperature.toDouble()).put("seed", 42).put("thinking", false)
                .put("ignore_eos", false)
            val power = app.getSystemService(PowerManager::class.java)
            val thermalBefore = if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null
            val output = ByteArrayOutputStream()
            var lastUpdate = 0L
            var lastSave = SystemClock.elapsedRealtime()
            var tokens = 0
            val result = try {
                NativeBridge.generate(request.toString().toByteArray(), ByteSink { chunk ->
                    output.write(chunk)
                    tokens++
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastUpdate >= UPDATE_INTERVAL_MS) {
                        lastUpdate = now
                        // Partial answers reach disk every few seconds, so a killed process loses little.
                        val save = now - lastSave >= 3000
                        if (save) lastSave = now
                        val partial = decodeComplete(output.toByteArray())
                        ChatStore.updateMessage(chat.id, answerId, save) { it.copy(text = partial) }
                        update { s -> s.copy(status = "Writing…", generation = s.generation?.copy(tokens = tokens)) }
                    }
                })
            } catch (error: Exception) {
                val failure = error.message ?: error.javaClass.simpleName
                ChatStore.updateMessage(chat.id, answerId, true) {
                    it.copy(streaming = false, failed = true,
                        text = decodeComplete(output.toByteArray()).ifEmpty { "Generation failed." })
                }
                update { s -> s.copy(phase = Phase.Ready, status = "Ready", error = failure, generation = null) }
                if (!AppVisibility.visible) Notifications.answerFinished(app, chat.id, chat.title, failure, failed = true)
                return@task
            }
            val json = JSONObject(result.decodeToString())
            json.put("surface", "app").put("model_sha256", modelHash ?: JSONObject.NULL)
                .put("workload_sha256", MessageDigest.getInstance("SHA-256").digest(request.toString().toByteArray()).joinToString("") { "%02x".format(it) })
                .put("thermal_status_before", thermalBefore ?: JSONObject.NULL)
                .put("thermal_status_after", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else JSONObject.NULL)
                .put("finished_in_background", !AppVisibility.visible)
            val answer = json.getString("text").ifEmpty { "(no answer)" }
            val termination = json.getString("termination")
            val speed = json.getDouble("decode_tokens_per_second")
            val stats = buildString {
                append("%.1f tok/s".format(Locale.US, speed))
                append(" · ${json.getInt("generated_tokens")} tokens")
                if (!json.isNull("ttft_ms")) append(" · first token %.1f s".format(Locale.US, json.getDouble("ttft_ms") / 1000))
                if (termination == "cancelled") append(" · stopped")
                if (termination == "length") append(" · length limit")
            }
            val stopped = termination == "cancelled"
            ChatStore.updateMessage(chat.id, answerId, true) { it.copy(text = answer, streaming = false, stats = stats, stopped = stopped) }
            update { s ->
                s.copy(phase = Phase.Ready, status = "Ready", lastResult = json.toString(2), lastSpeed = speed, generation = null)
            }
            if (!AppVisibility.visible) Notifications.answerFinished(app, chat.id, chat.title, answer, failed = false, stopped = stopped)
        }
        return true
    }

    /**
     * The conversation as the engine sees it. A turn is context only once its answer completed; the newest user
     * message is the question. Attached documents become text; images are files the engine encodes.
     */
    private fun requestMessages(chat: Chat, vision: Boolean): List<JSONObject> {
        val list = chat.messages.dropLast(1)
        val turns = mutableListOf<Message>()
        for ((i, message) in list.withIndex()) {
            if (!message.fromUser) continue
            if (i == list.lastIndex) {
                turns += message
                continue
            }
            val answer = list[i + 1]
            if (!answer.fromUser && !answer.failed && !answer.stopped && !answer.streaming) turns += listOf(message, answer)
        }
        val shown = if (!vision) emptySet() else turns.asReversed().asSequence().filter { it.fromUser }
            .flatMap { m -> m.attachments.asReversed().asSequence().filter { it.kind == AttachmentKind.Image }.map { it.file } }
            .take(MAX_CONTEXT_IMAGES).toSet()
        return turns.map { m ->
            // Text-only messages keep plain string content, exactly as before attachments existed.
            val content: Any = if (!m.fromUser || m.attachments.isEmpty()) m.text else {
                val parts = JSONArray()
                val text = StringBuilder()
                for (attachment in m.attachments) when (attachment.kind) {
                    AttachmentKind.Image -> when {
                        attachment.file in shown -> parts.put(JSONObject().put("type", "image")
                            .put("path", ChatStore.file(chat.id, attachment).absolutePath))
                        vision -> text.append("[An earlier image, \"${attachment.name}\", is no longer shown.]\n\n")
                        else -> text.append("[The user attached an image, \"${attachment.name}\", that this model cannot see.]\n\n")
                    }
                    AttachmentKind.Document -> {
                        val body = runCatching { ChatStore.file(chat.id, attachment).readText() }.getOrDefault("(unreadable)")
                        text.append("[File: ${attachment.name}]\n").append(body).append("\n[End of file]\n\n")
                    }
                }
                text.append(m.text)
                parts.put(JSONObject().put("type", "text").put("text", text.toString().trim()))
            }
            JSONObject().put("role", if (m.fromUser) "user" else "assistant").put("content", content)
        }
    }

    /** Decodes UTF-8 without emitting a partial trailing code point. */
    private fun decodeComplete(bytes: ByteArray): String {
        val decoder = Charsets.UTF_8.newDecoder()
        val chars = CharBuffer.allocate(bytes.size)
        decoder.decode(ByteBuffer.wrap(bytes), chars, false)
        chars.flip()
        return chars.toString()
    }
}
