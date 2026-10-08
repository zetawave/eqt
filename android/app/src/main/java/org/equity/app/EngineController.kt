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
    val threads: Int = 4,
    val context: Int = 2048,
    /** 0 selects the expert-cache budget from available RAM at load time. */
    val expertCacheMiB: Int = 0,
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
        )
    }

    fun update(context: Context, change: (Settings) -> Settings) {
        val next = change(flow.value)
        flow.value = next
        context.getSharedPreferences("settings", Context.MODE_PRIVATE).edit()
            .putString("system", next.systemPrompt).putInt("max_tokens", next.maxTokens)
            .putFloat("temperature", next.temperature).putInt("threads", next.threads)
            .putInt("context", next.context).putInt("expert_cache", next.expertCacheMiB).apply()
    }
}

enum class Phase { Empty, Loading, Ready, Generating, Unloading }

data class Message(
    val id: Long,
    val fromUser: Boolean,
    val text: String,
    val streaming: Boolean = false,
    val stats: String? = null,
    val failed: Boolean = false,
)

data class EngineState(
    val phase: Phase = Phase.Empty,
    val modelId: String? = null,
    val modelName: String? = null,
    val status: String = "No model loaded",
    val loadPercent: Int? = null,
    val configuration: String? = null,
    val messages: List<Message> = emptyList(),
    val lastResult: String? = null,
    val lastSpeed: Double? = null,
    val error: String? = null,
)

/** Owns the single native engine. Native calls run on one worker thread; only cancellation crosses threads. */
object EngineController {
    private val worker = Executors.newSingleThreadExecutor()
    private val flow = MutableStateFlow(EngineState())
    val state: StateFlow<EngineState> = flow.asStateFlow()
    private var descriptor: ParcelFileDescriptor? = null
    private var history = JSONArray()
    private var modelHash: String? = null
    private var nextId = 1L
    @Volatile private var cancelled = false

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
        startLoad(model.id, model.name)
        task {
            guardLoad {
                val path = ModelFiles.final(app, model)
                require(ModelFiles.installed(app, model)) { "${model.name} is not downloaded" }
                val options = loadOptions(app, model.streamed, model.residentMiB, model.contextMiBPer1k, path.length(), settings)
                loadNative(path.absolutePath, options, model.sha256)
            }
        }
    }

    /** Loads a user-selected local GGUF through a seekable descriptor, without copying it. */
    fun load(context: Context, uri: Uri) {
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        startLoad(null, "Local model")
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
                val options = loadOptions(app, streamed, if (streamed) 2448 else sizeMiB, 112, stat.st_size, settings)
                loadNative("/proc/self/fd/${pfd.fd}", options, hash)
            }
        }
    }

    private fun startLoad(id: String?, name: String) {
        update {
            it.copy(phase = Phase.Loading, modelId = id, modelName = name, loadPercent = null, error = null,
                status = "Preparing $name…", configuration = null)
        }
    }

    /** Any load failure returns the engine to an empty, consistent state. */
    private fun guardLoad(action: () -> Unit) {
        try {
            action()
        } catch (error: Exception) {
            runCatching { NativeBridge.unload() }
            closeDescriptor()
            update {
                EngineState(status = "No model loaded",
                    error = if (cancelled) null else (error.message ?: error.javaClass.simpleName))
            }
        }
    }

    private fun availableMiB(context: Context): Int {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        return (memory.availMem shr 20).toInt()
    }

    private fun loadOptions(context: Context, streamed: Boolean, residentMiB: Int, contextMiBPer1k: Int, fileBytes: Long,
                            settings: Settings): JSONObject {
        val available = availableMiB(context)
        // Context state and compute buffers are allocated at load; admitting without them invites the low-memory killer.
        // Prompt batches of 512 tokens take the tiled GEMM path and need larger compute buffers.
        val workMiB = contextMiBPer1k * settings.context / 1024 + 512
        val options = JSONObject().put("context", settings.context).put("batch", 512).put("threads", settings.threads)
        if (!streamed) {
            val fileMiB = (fileBytes shr 20).toInt()
            require(fileMiB + workMiB + 512 <= available) {
                "This model needs about ${fileMiB + workMiB + 512} MiB of free memory and $available MiB are available. " +
                    "Close some apps or choose a smaller context in Settings."
            }
            return options.put("mmap", true).put("memory_budget_mib", maxOf(2048, fileMiB + 1024))
        }
        // Expert cache from free RAM after shared weights, compute buffers, the transient next-layer allowance used
        // while reading prompts, and system headroom.
        val prefillMiB = 512
        val cache = settings.expertCacheMiB.takeIf { it > 0 }
            ?: (available - residentMiB - workMiB - prefillMiB - 1024).coerceIn(512, 3072)
        require(residentMiB + cache + prefillMiB + workMiB + 512 <= available) {
            "This model needs about ${residentMiB + cache + prefillMiB + workMiB + 512} MiB of free memory and " +
                "$available MiB are available. Close some apps and try again."
        }
        // Measured defaults (docs/decisions.md D018, D020): coalesced expert reads, next-layer prefetch for prompts;
        // the engine keeps experts in reusable slots by default.
        return options.put("mmap", false).put("expert_streaming", true).put("expert_cache_mib", cache)
            .put("io_threads", 16).put("direct_io", true).put("prefetch", true)
            .put("coalesce_kib", 2048).put("coalesce_gap", 1).put("prefill_prefetch_mib", prefillMiB)
            .put("memory_budget_mib", residentMiB + cache + prefillMiB + 1536)
    }

    private fun loadNative(path: String, options: JSONObject, hash: String?) {
        NativeBridge.unload()
        val progress = ByteSink { bytes ->
            val percent = bytes.decodeToString().toIntOrNull()
            update { it.copy(loadPercent = percent, status = "Loading ${percent ?: 0}%") }
        }
        val info = try {
            JSONObject(NativeBridge.load(path.toByteArray(), options.toString().toByteArray(), progress).decodeToString())
        } catch (error: IllegalStateException) {
            // Storage without O_DIRECT support falls back to buffered reads, and says so.
            if (!options.optBoolean("direct_io") || error.message?.contains("O_DIRECT") != true) throw error
            options.put("direct_io", false)
            JSONObject(NativeBridge.load(path.toByteArray(), options.toString().toByteArray(), progress).decodeToString())
        }
        modelHash = hash
        history = JSONArray()
        val streaming = info.optJSONObject("expert_streaming")
        val configuration = buildString {
            append("CPU ").append(info.optString("cpu_variant").removePrefix("android_"))
            append(" · ${info.getInt("threads")} threads · ${info.getInt("context")} context")
            if (streaming != null) {
                append(" · ${streaming.getLong("effective_budget_bytes") shr 20} MiB expert cache")
                if (!options.optBoolean("direct_io")) append(" · buffered reads")
            }
        }
        update {
            it.copy(phase = Phase.Ready, loadPercent = null, messages = emptyList(), lastResult = null, lastSpeed = null,
                configuration = configuration, status = "Ready")
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
            history = JSONArray()
            update { EngineState(status = "Model unloaded") }
        }
    }

    /** Cancels loading or generation; safe from any thread. */
    fun stop() {
        cancelled = true
        NativeBridge.cancel()
    }

    fun memoryPressure() {
        if (flow.value.phase == Phase.Empty) return
        stop()
        task {
            NativeBridge.unload()
            closeDescriptor()
            history = JSONArray()
            update { EngineState(status = "No model loaded", error = "Android asked to free memory, so the model was unloaded.") }
        }
    }

    fun newChat() {
        if (flow.value.phase == Phase.Generating) return
        history = JSONArray()
        update { it.copy(messages = emptyList()) }
    }

    fun send(context: Context, prompt: String) {
        val text = prompt.trim()
        if (flow.value.phase != Phase.Ready || text.isEmpty()) return
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        val answerId = nextId + 1
        update {
            it.copy(phase = Phase.Generating, status = "Reading your message…", error = null,
                messages = it.messages + Message(nextId, true, text) + Message(answerId, false, "", streaming = true))
        }
        nextId += 2
        task {
            val messages = JSONArray().put(JSONObject().put("role", "system").put("content", settings.systemPrompt))
            for (i in 0 until history.length()) messages.put(history.getJSONObject(i))
            messages.put(JSONObject().put("role", "user").put("content", text))
            val request = JSONObject().put("messages", messages).put("max_tokens", settings.maxTokens)
                .put("temperature", settings.temperature.toDouble()).put("seed", 42).put("thinking", false)
                .put("ignore_eos", false)
            val power = app.getSystemService(PowerManager::class.java)
            val thermalBefore = if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null
            val output = ByteArrayOutputStream()
            var lastUpdate = 0L
            val result = try {
                NativeBridge.generate(request.toString().toByteArray(), ByteSink { chunk ->
                    output.write(chunk)
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastUpdate >= 60) {
                        lastUpdate = now
                        val partial = decodeComplete(output.toByteArray())
                        update { s -> s.copy(status = "Writing…", messages = s.messages.map { m -> if (m.id == answerId) m.copy(text = partial) else m }) }
                    }
                })
            } catch (error: Exception) {
                update { s ->
                    s.copy(phase = Phase.Ready, status = "Ready", error = error.message,
                        messages = s.messages.map { m ->
                            if (m.id == answerId) m.copy(streaming = false, failed = true, text = m.text.ifEmpty { "Generation failed." }) else m
                        })
                }
                return@task
            }
            val json = JSONObject(result.decodeToString())
            json.put("surface", "app").put("model_sha256", modelHash ?: JSONObject.NULL)
                .put("workload_sha256", MessageDigest.getInstance("SHA-256").digest(request.toString().toByteArray()).joinToString("") { "%02x".format(it) })
                .put("thermal_status_before", thermalBefore ?: JSONObject.NULL)
                .put("thermal_status_after", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else JSONObject.NULL)
            val answer = json.getString("text")
            val termination = json.getString("termination")
            if (termination != "cancelled") {
                history.put(JSONObject().put("role", "user").put("content", text))
                history.put(JSONObject().put("role", "assistant").put("content", answer))
            }
            val speed = json.getDouble("decode_tokens_per_second")
            val stats = buildString {
                append("%.1f tok/s".format(Locale.US, speed))
                append(" · ${json.getInt("generated_tokens")} tokens")
                if (!json.isNull("ttft_ms")) append(" · first token %.1f s".format(Locale.US, json.getDouble("ttft_ms") / 1000))
                if (termination == "cancelled") append(" · stopped")
                if (termination == "length") append(" · length limit")
            }
            update { s ->
                s.copy(phase = Phase.Ready, status = "Ready", lastResult = json.toString(2), lastSpeed = speed,
                    messages = s.messages.map { m ->
                        if (m.id == answerId) m.copy(text = answer.ifEmpty { "(no answer)" }, streaming = false, stats = stats) else m
                    })
            }
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
