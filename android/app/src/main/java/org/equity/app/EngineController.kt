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
import java.io.File
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
    /** The model reasons before answering, when its chat template supports it. */
    val thinking: Boolean = false,
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
            thinking = p.getBoolean("thinking", d.thinking),
        )
    }

    fun update(context: Context, change: (Settings) -> Settings) {
        val next = change(flow.value)
        flow.value = next
        context.getSharedPreferences("settings", Context.MODE_PRIVATE).edit()
            .putString("system", next.systemPrompt).putInt("max_tokens", next.maxTokens)
            .putFloat("temperature", next.temperature).putInt("threads", next.threads)
            .putInt("context", next.context).putInt("expert_cache", next.expertCacheMiB)
            .putBoolean("gpu", next.gpu).putBoolean("thinking", next.thinking).apply()
    }
}

enum class Phase { Empty, Loading, Ready, Generating, Unloading }

/** The answer being written: its chat, its message and how many tokens have arrived. */
data class Generation(val chatId: String, val answerId: Long, val startedAt: Long, val tokens: Int = 0,
                      val hasVisibleText: Boolean = false, val thinking: Boolean = false)

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
    val modelSha256: String? = null,
    val architecture: String? = null,
    val embeddingSize: Int? = null,
    val decoderBlocks: Int? = null,
    /** The loaded model's chat template has a thinking mode. */
    val thinkingSupported: Boolean = false,
    val steeringProfile: SteeringProfile? = null,
    val steeringNote: String? = null,
    val steeringImportMessage: String? = null,
    /** The model fingerprint [steeringImportMessage] refers to, so its card shows it. */
    val steeringMessageModel: String? = null,
    /** Incremented after every import or removal, so per-model summaries are read again. */
    val steeringRevision: Int = 0,
    val steeringOperation: Boolean = false,
) {
    val busy get() = steeringOperation || phase == Phase.Loading || phase == Phase.Generating || phase == Phase.Unloading
}

/** Owns the single native engine. Native calls run on one worker thread; only cancellation crosses threads. */
object EngineController {
    private val worker = Executors.newSingleThreadExecutor()
    private val flow = MutableStateFlow(EngineState())
    val state: StateFlow<EngineState> = flow.asStateFlow()
    private var descriptor: ParcelFileDescriptor? = null
    private var modelHash: String? = null
    private data class ModelSource(
        val path: String,
        val hash: String?,
        val streamed: Boolean,
        val residentMiB: Int,
        val contextMiBPer1k: Int,
        val fileBytes: Long,
        val settings: Settings,
        val projectorPath: String? = null,
        val visionMiB: Int = 0,
        val stream: StreamProfile = StreamProfile(),
    )
    private var loadedSource: ModelSource? = null
    private var trainedContext: Int? = null
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

    private fun task(prepare: Boolean = true, action: () -> Unit) {
        if (prepare) {
            cancelled = false
            NativeBridge.prepare()
        }
        worker.execute {
            try {
                action()
            } catch (error: Exception) {
                update { it.copy(error = error.message ?: error.javaClass.simpleName) }
            }
        }
    }

    fun clearError(message: String? = null) = update {
        if (message == null || it.error == message) it.copy(error = null) else it
    }

    private fun refreshSteering(context: Context) {
        val hash = modelHash
        if (hash == null) {
            update { it.copy(modelSha256 = null, steeringProfile = null,
                steeringNote = if (loadedSource == null) "Load the matching model to use its vector." else "Verify this model to find compatible vectors.") }
            return
        }
        try {
            val profile = SteeringFiles.read(context, hash)
            val engine = flow.value
            require(profile == null || (profile.architecture == engine.architecture && profile.embedding == engine.embeddingSize &&
                profile.layer < (engine.decoderBlocks ?: 0))) { "Imported steering dimensions do not match this model" }
            update { it.copy(modelSha256 = hash, steeringProfile = profile,
                steeringNote = if (profile == null) "No vector imported for this model." else null) }
        } catch (error: Exception) {
            update { it.copy(modelSha256 = hash, steeringProfile = null, steeringNote = "Steering unavailable: ${error.message}") }
        }
    }

    /** Imports a vector for the model whose fingerprint is [modelHash] (null: whichever model the files name). */
    fun importSteering(context: Context, uris: List<Uri>, modelHash: String? = null) {
        val previous = flow.value
        if (previous.busy) return
        val app = context.applicationContext
        update { it.copy(phase = Phase.Loading, loadPercent = null, status = "Importing steering vector…", error = null,
            steeringImportMessage = null, steeringMessageModel = modelHash, steeringOperation = true) }
        task {
            try {
                val imported = SteeringFiles.importVector(app, uris, modelHash) { cancelled }
                refreshSteering(app)
                val loaded = imported.modelHash == flow.value.modelSha256
                update { it.copy(steeringMessageModel = imported.modelHash,
                    steeringImportMessage = if (loaded) "Vector imported. Steering is available in this model's chats."
                        else "Vector imported. Steering appears in chats while this model is loaded.") }
            } catch (error: Exception) {
                if (!cancelled) update { it.copy(error = error.message ?: "Steering import failed") }
            } finally {
                update { it.copy(steeringRevision = it.steeringRevision + 1) }
                finishSteeringOperation(previous)
            }
        }
    }

    fun removeSteering(context: Context, modelHash: String) {
        val previous = flow.value
        if (previous.busy) return
        update { it.copy(phase = Phase.Loading, steeringOperation = true, status = "Removing steering vector…",
            steeringImportMessage = null) }
        task {
            try {
                SteeringFiles.remove(context.applicationContext, modelHash)
                refreshSteering(context.applicationContext)
            } finally {
                update { it.copy(steeringRevision = it.steeringRevision + 1) }
                finishSteeringOperation(previous)
            }
        }
    }

    private fun finishSteeringOperation(previous: EngineState) = update {
        val next = if (it.phase == Phase.Loading) it.copy(phase = previous.phase, status = previous.status) else it
        next.copy(loadPercent = null, steeringOperation = false)
    }

    /** Explicitly hashes a large picker model once; catalog models already have a verified fingerprint. */
    fun verifySteeringModel(context: Context) {
        if (flow.value.phase != Phase.Ready) return
        val source = loadedSource ?: return
        update { it.copy(phase = Phase.Loading, loadPercent = null, status = "Verifying model for steering…", error = null,
            steeringOperation = true) }
        task {
            try {
                val before = Os.stat(source.path)
                val digest = MessageDigest.getInstance("SHA-256")
                File(source.path).inputStream().use { input ->
                    val buffer = ByteArray(1024 * 1024)
                    var total = 0L
                    while (true) {
                        check(!cancelled) { "Model verification cancelled" }
                        val count = input.read(buffer)
                        if (count < 0) break
                        digest.update(buffer, 0, count)
                        total += count
                    }
                    require(total == source.fileBytes) { "Model size changed" }
                }
                val after = Os.stat(source.path)
                require(before.st_size == after.st_size && before.st_mtime == after.st_mtime) { "Model changed during verification" }
                val hash = digest.digest().joinToString("") { "%02x".format(it) }
                // Reload shared/resident tensors too, so the fingerprint describes the weights actually in use.
                guardLoad {
                    loadSource(context.applicationContext, source.copy(hash = hash))
                    val loaded = Os.stat(source.path)
                    require(loaded.st_size == before.st_size && loaded.st_mtime == before.st_mtime) {
                        "Model changed while reloading verified weights"
                    }
                }
            } catch (error: Exception) {
                if (!cancelled) update { it.copy(error = error.message ?: "Model verification failed") }
            } finally {
                update { it.copy(phase = if (loadedSource == null) Phase.Empty else Phase.Ready, loadPercent = null,
                    status = if (loadedSource == null) "No model loaded" else "Ready", steeringOperation = false) }
            }
        }
    }

    /** Loads a downloaded catalog model with options tuned for it. */
    fun load(context: Context, model: CatalogModel) {
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        startLoad(app, model.id, model.name)
        task {
            guardLoad {
                NativeBridge.unload()
                closeDescriptor()
                val path = ModelFiles.final(app, model.artifact)
                require(ModelFiles.installed(app, model.artifact)) { "${model.name} is not downloaded" }
                val projector = model.projector?.takeIf { ModelFiles.installed(app, it) }
                val visionMiB = projector?.let { (it.sizeBytes shr 20).toInt() + VISION_WORK_MIB } ?: 0
                loadSource(app, ModelSource(path.absolutePath, model.sha256, model.streamed, model.residentMiB,
                    model.contextMiBPer1k, path.length(), settings,
                    projector?.let { ModelFiles.final(app, it).absolutePath }, visionMiB, model.stream))
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
                NativeBridge.unload()
                closeDescriptor()
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
                // A streamed file of a catalog model's exact size is admitted with that model's memory figures (its
                // hash is not checked, so it gets no fingerprint); other files use the larger per-token context cost
                // of the catalog models.
                val sized = Catalog.models.firstOrNull { it.streamed && streamed && it.sizeBytes == stat.st_size }
                loadSource(app, ModelSource("/proc/self/fd/${pfd.fd}", hash, streamed,
                    sized?.residentMiB ?: if (streamed) 2448 else sizeMiB, sized?.contextMiBPer1k ?: 112, stat.st_size,
                    settings, stream = sized?.stream ?: StreamProfile()))
            }
        }
    }

    private fun startLoad(context: Context, id: String?, name: String) {
        update {
            it.copy(phase = Phase.Loading, modelId = id, modelName = name, loadPercent = null, error = null,
                status = "Preparing $name…", configuration = null, vision = false, lastModelId = id ?: it.lastModelId,
                modelSha256 = null, steeringProfile = null, steeringNote = null)
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

    /** One way to read prompts for a streamed model, and the free memory it needs. */
    private class StreamPlan(val batch: Int, val prefillMiB: Int, val cacheMiB: Int, val shortfallMiB: Int, val requiredMiB: Int)

    /**
     * Memory for reading prompts in [batch]-token batches with a [prefillMiB] next-layer allowance. With [available]
     * the expert cache takes what is left (between its floor and 3 GiB); without it the cache is at its floor, which is
     * the least the model needs.
     */
    private fun streamPlan(available: Int?, residentMiB: Int, kvMiB: Int, settings: Settings, visionMiB: Int,
                           stream: StreamProfile, batch: Int, prefillMiB: Int): StreamPlan {
        val gpuStagingMiB = if (settings.gpu) GPU_STAGING_MIB else 0
        val workMiB = if (batch == 2048) kvMiB + 512 + GPU_WORK_MIB else kvMiB + stream.computeMiBPer512 * batch / 512
        val layerMiB = stream.layerWorkingSetMiB(batch)
        // Images are encoded before the prompt is read and the projector is released before it, so image memory and
        // the prompt-time prefetch allowance are never needed at the same time.
        val transientMiB = maxOf(prefillMiB + gpuStagingMiB, visionMiB)
        val floor = maxOf(512, layerMiB)
        val cache = settings.expertCacheMiB.takeIf { it > 0 }
            ?: available?.let { (it - residentMiB - workMiB - transientMiB - SYSTEM_RESERVE_MIB).coerceIn(floor, 3072) }
            ?: floor
        val shortfallMiB = maxOf(0, layerMiB - cache)
        return StreamPlan(batch, prefillMiB, cache, shortfallMiB,
            residentMiB + cache + shortfallMiB + transientMiB + workMiB + SYSTEM_RESERVE_MIB)
    }

    /** Free memory, in MiB, that [model] needs to load with full prompt batches and the next-layer allowance. */
    fun fullSpeedMemoryMiB(model: CatalogModel): Int {
        val settings = SettingsStore.state.value
        val kvMiB = model.contextMiBPer1k * settings.context / 1024
        if (!model.streamed) return (model.sizeBytes shr 20).toInt() + kvMiB + 512 + 512
        return streamPlan(null, model.residentMiB, kvMiB, settings, 0, model.stream,
            if (settings.gpu) 2048 else 512, 512).requiredMiB
    }

    fun availableMiB(context: Context): Int {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        return (memory.availMem shr 20).toInt()
    }

    /** [visionMiB] is memory kept free for image encoding; 0 for a text-only load. */
    private fun loadOptions(context: Context, streamed: Boolean, residentMiB: Int, contextMiBPer1k: Int, fileBytes: Long,
                            settings: Settings, visionMiB: Int, stream: StreamProfile): JSONObject {
        val available = availableMiB(context)
        // Context state and compute buffers are allocated at load; admitting without them invites the low-memory killer.
        // Prompt batches of 512 tokens take the tiled GEMM path and need larger compute buffers.
        // GPU mode reads prompts in batches of 2048 tokens (fewer passes over the streamed experts, fuller GPU
        // tiles): larger compute buffers, plus staging for one GPU operation that is released after each prompt.
        val gpuWorkMiB = if (settings.gpu) GPU_WORK_MIB else 0
        val gpuStagingMiB = if (settings.gpu) GPU_STAGING_MIB else 0
        val kvMiB = contextMiBPer1k * settings.context / 1024
        val workMiB = kvMiB + 512 + gpuWorkMiB
        val firstBatch = if (settings.gpu) 2048 else 512
        val options = JSONObject().put("context", settings.context).put("batch", firstBatch)
            .put("threads", settings.threads)
        if (settings.gpu) options.put("gpu", true)
        if (!streamed) {
            val fileMiB = (fileBytes shr 20).toInt()
            val requiredMiB = fileMiB + workMiB + 512 + maxOf(visionMiB, gpuStagingMiB)
            require(requiredMiB <= available) {
                "This model needs about $requiredMiB MiB of free memory and $available MiB are available. " +
                    "Close some apps or choose a smaller context in Settings."
            }
            return options.put("mmap", true).put("memory_budget_mib", maxOf(2048, requiredMiB))
        }
        // Expert cache from free RAM after shared weights, compute buffers, the transient next-layer allowance used
        // while reading prompts, and system headroom. A prompt batch keeps its share of one layer's experts resident
        // at once and the engine fits them inside the cache (D026), so the cache never starts below that share and a
        // smaller fixed cache needs the shortfall on top. When memory is short, prompts are first read without the
        // next-layer allowance, then in smaller batches: fewer of a layer's experts and smaller compute buffers at once
        // (Flash-Next at 128 tokens: 843 instead of 916 MiB of experts, 135 instead of 540 MiB of buffers). Answers
        // stay the same up to float rounding; prompts longer than the batch take more passes over the experts.
        val plans = listOf(firstBatch to 512, firstBatch to 0) + listOf(256, 128, 64, 32).filter { it < firstBatch }.map { it to 0 }
        var required = 0
        for ((batch, prefillMiB) in plans) {
            val plan = streamPlan(available, residentMiB, kvMiB, settings, visionMiB, stream, batch, prefillMiB)
            required = plan.requiredMiB
            if (required > available) continue
            // Measured defaults (docs/decisions.md D018, D020): coalesced expert reads, next-layer prefetch for
            // prompts; the engine keeps experts in reusable slots by default.
            return options.put("batch", batch).put("mmap", false).put("expert_streaming", true)
                .put("expert_cache_mib", plan.cacheMiB)
                .put("io_threads", 16).put("direct_io", true).put("prefetch", true)
                .put("coalesce_kib", 2048).put("coalesce_gap", 1).put("prefill_prefetch_mib", prefillMiB)
                .put("memory_budget_mib", residentMiB + plan.cacheMiB + plan.shortfallMiB + prefillMiB + 1536 +
                    gpuWorkMiB + gpuStagingMiB + kvMiB)
                .apply { stream.tuning.forEach { (key, value) -> put(key, value) } }
        }
        throw IllegalArgumentException("This model needs about $required MiB of free memory and $available MiB are " +
            "available. Close some apps or choose a smaller context in Settings, then try again.")
    }

    private fun loadSource(context: Context, source: ModelSource, growing: Boolean = false) {
        // Reclaim the previous context and expert cache before sizing their replacements from free memory.
        NativeBridge.unload()
        loadedSource = null
        fun options(visionMiB: Int) = loadOptions(context, source.streamed, source.residentMiB,
            source.contextMiBPer1k, source.fileBytes, source.settings, visionMiB, source.stream)
        val withVision = source.projectorPath?.let { path ->
            if (growing) options(source.visionMiB).put("mmproj", path)
            else runCatching { options(source.visionMiB).put("mmproj", path) }.getOrNull()
        }
        val options = withVision ?: options(0)
        source.hash?.let { options.put("model_sha256", it) }
        // An expert pack beside the model (eqt-pack, D032) holds its experts one contiguous read each; after
        // `eqt-pack move` the model file has no experts of its own and the engine refuses it without the pack.
        if (source.streamed) {
            File(source.path.removeSuffix(".gguf") + ".eqtpack").takeIf { it.isFile }?.let { options.put("expert_pack", it.path) }
        }
        val note = if (source.projectorPath != null && withVision == null) "images off: not enough free memory" else null
        loadNative(source.path, options, source.hash, note, growing)
        loadedSource = source.copy(settings = source.settings.copy(gpu = options.optBoolean("gpu")),
            projectorPath = source.projectorPath.takeIf { withVision != null })
        refreshSteering(context)
    }

    private fun loadNative(path: String, options: JSONObject, hash: String?, note: String?, growing: Boolean) {
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
                    // A batch the memory fit already lowered stays lowered.
                    options.put("batch", minOf(512, options.getInt("batch")))
                    gpuNote = "GPU unavailable"
                }
                else -> throw error
            }
            JSONObject(NativeBridge.load(path.toByteArray(), options.toString().toByteArray(), progress).decodeToString())
        }
        modelHash = hash
        trainedContext = if (info.isNull("trained_context")) null else info.getInt("trained_context")
        val vision = !info.isNull("vision")
        val streaming = info.optJSONObject("expert_streaming")
        val configuration = buildString {
            append("CPU ").append(info.optString("cpu_variant").removePrefix("android_"))
            info.optJSONObject("gpu")?.let { append(" + GPU ").append(it.optString("device")) }
            append(" · ${info.getInt("threads")} threads · ${info.getInt("context")} context")
            if (info.getInt("batch") < 512) append(" · ${info.getInt("batch")}-token prompt batches")
            if (streaming != null) {
                append(" · ${streaming.getLong("effective_budget_bytes") shr 20} MiB expert cache")
                if (!options.optBoolean("direct_io")) append(" · buffered reads")
            }
            if (vision) append(" · images")
            note?.let { append(" · ").append(it) }
            gpuNote?.let { append(" · ").append(it) }
        }
        update {
            it.copy(phase = if (growing) Phase.Generating else Phase.Ready, loadPercent = null,
                lastResult = null, lastSpeed = null, configuration = configuration,
                status = if (growing) "Reading your message…" else "Ready", vision = vision)
                .copy(modelSha256 = hash, architecture = info.getString("architecture"),
                    embeddingSize = info.getInt("embedding_size"), decoderBlocks = info.getInt("decoder_blocks"),
                    thinkingSupported = info.optBoolean("thinking_supported"))
        }
    }

    private fun generateWithContext(context: Context, request: JSONObject, sink: ByteSink): ByteArray {
        val bytes = request.toString().toByteArray()
        try {
            return NativeBridge.generate(bytes, sink)
        } catch (overflow: ContextOverflowException) {
            val source = loadedSource ?: throw overflow
            val required = overflow.promptTokens + overflow.outputTokens
            val limit = trainedContext ?: (1 shl 20)
            require(required <= limit) {
                "This message and its answer need $required tokens, exceeding the model's $limit-token context. " +
                    "Use a smaller attachment or start a new chat."
            }
            val expanded = (((required + 2047) / 2048) * 2048).coerceAtMost(limit)
            check(!cancelled) { "Generation cancelled" }
            update { it.copy(status = "Expanding context to $expanded tokens…") }
            try {
                loadSource(context, source.copy(settings = source.settings.copy(context = expanded)), growing = true)
            } catch (error: Exception) {
                runCatching { NativeBridge.unload() }
                closeDescriptor()
                throw IllegalStateException("This message needs $required context tokens. " +
                    "Could not allocate a larger context: ${error.message}", error)
            }
            check(!cancelled) { "Generation cancelled" }
            // Overflow is reported before evaluation, so retrying preserves the full prompt and sampling request.
            return NativeBridge.generate(bytes, sink)
        }
    }

    private fun closeDescriptor() {
        descriptor?.close()
        descriptor = null
        loadedSource = null
        trainedContext = null
        modelHash = null
    }

    fun unload() {
        if (flow.value.phase == Phase.Empty) return
        stop()
        update { it.copy(phase = Phase.Unloading, status = "Unloading…") }
        task(prepare = false) {
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
        task(prepare = false) {
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
        if (engine.phase != Phase.Ready || engine.steeringOperation || (text.isEmpty() && attachments.isEmpty())) return false
        if (attachments.any { it.kind == AttachmentKind.Image } && !engine.vision) return false
        // A chat's steering applies while a model with an imported vector is loaded; chats offer it only then, and
        // the answer's details show whether it was applied.
        val chatSteering = ChatStore.state.value.steering
        val steering = chatSteering.takeIf { engine.steeringProfile != null && engine.modelSha256 != null } ?: SteeringOptions()
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        val thinking = settings.thinking && engine.thinkingSupported
        val (chat, answerId) = ChatStore.startTurn(chatId, text, attachments)
        update {
            it.copy(phase = Phase.Generating, status = "Reading your message…", error = null,
                generation = Generation(chat.id, answerId, SystemClock.elapsedRealtime(), thinking = thinking))
        }
        InferenceService.start(app)
        task {
            val request = JSONObject()
            val power = app.getSystemService(PowerManager::class.java)
            val thermalBefore = if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null
            val output = ByteArrayOutputStream()
            var lastUpdate = 0L
            var lastSave = SystemClock.elapsedRealtime()
            var tokens = 0
            // Live reasoning time, from the first generated token to the closing tag (to update granularity);
            // the engine's measurement replaces it when the answer is complete.
            var firstTokenAt = 0L
            var reasoningMs: Long? = null
            val result = try {
                val messages = JSONArray().put(JSONObject().put("role", "system").put("content", settings.systemPrompt))
                requestMessages(chat, engine.vision).forEach { messages.put(it) }
                request.put("messages", messages).put("max_tokens", settings.maxTokens)
                    .put("temperature", settings.temperature.toDouble()).put("seed", 42).put("thinking", thinking)
                    .put("ignore_eos", false)
                if (steering.enabled) {
                    request.put("model_sha256", engine.modelSha256)
                        .put("steering", requireNotNull(engine.steeringProfile).request(app, steering.strength))
                }
                generateWithContext(app, request, ByteSink { chunk ->
                    output.write(chunk)
                    tokens++
                    val now = SystemClock.elapsedRealtime()
                    if (tokens == 1) firstTokenAt = now
                    if (now - lastUpdate >= UPDATE_INTERVAL_MS) {
                        lastUpdate = now
                        // Partial answers reach disk every few seconds, so a killed process loses little.
                        val save = now - lastSave >= 3000
                        if (save) lastSave = now
                        val partial = decodeComplete(output.toByteArray())
                        val split = if (thinking) splitThinking(partial) else ThinkingSplit("", partial, true)
                        if (thinking && split.closed && reasoningMs == null) reasoningMs = now - firstTokenAt
                        val visible = split.answer.hasVisibleContent()
                        ChatStore.updateMessage(chat.id, answerId, save) {
                            it.copy(text = split.answer, reasoning = split.reasoning.takeIf { thinking }, reasoningMs = reasoningMs)
                        }
                        update { s -> s.copy(status = when {
                                !split.closed -> "Thinking…"
                                visible -> "Writing…"
                                else -> "Generating tokens; no visible text yet"
                            }, generation = s.generation?.copy(tokens = tokens, hasVisibleText = visible, thinking = !split.closed)) }
                    }
                })
            } catch (error: Exception) {
                val failure = error.message ?: error.javaClass.simpleName
                val partial = decodeComplete(output.toByteArray())
                val split = if (thinking) splitThinking(partial) else ThinkingSplit("", partial, true)
                ChatStore.updateMessage(chat.id, answerId, true) {
                    it.copy(streaming = false, failed = true, text = split.answer.ifEmpty { "Generation failed.\n\n$failure" },
                        reasoning = split.reasoning.takeIf { thinking && it.isNotEmpty() }, reasoningMs = reasoningMs,
                        stats = failure.takeIf { output.size() > 0 })
                }
                update { s ->
                    if (loadedSource == null) EngineState(lastModelId = s.lastModelId, error = failure)
                    else s.copy(phase = Phase.Ready, status = "Ready", error = failure, generation = null)
                }
                if (!AppVisibility.visible) Notifications.answerFinished(app, chat.id, chat.title, failure, failed = true)
                return@task
            }
            val json = JSONObject(result.decodeToString())
            json.put("surface", "app").put("model_sha256", modelHash ?: JSONObject.NULL)
                .put("workload_sha256", MessageDigest.getInstance("SHA-256").digest(request.toString().toByteArray()).joinToString("") { "%02x".format(it) })
                .put("thermal_status_before", thermalBefore ?: JSONObject.NULL)
                .put("thermal_status_after", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else JSONObject.NULL)
                .put("finished_in_background", !AppVisibility.visible)
            val split = if (thinking) splitThinking(json.getString("text")) else null
            // Reasoning that used every token leaves no answer; the message explains that instead of a placeholder.
            val answer = split?.answer ?: json.getString("text").ifEmpty { "(no answer)" }
            val thoughtMs = if (thinking) json.getDouble("reasoning_ms").toLong() else null
            val termination = json.getString("termination")
            val speed = json.getDouble("decode_tokens_per_second")
            val stats = buildString {
                append("%.1f tok/s".format(Locale.US, speed))
                append(" · ${json.getInt("generated_tokens")} tokens")
                if (!json.isNull("ttft_ms")) append(" · first token %.1f s".format(Locale.US, json.getDouble("ttft_ms") / 1000))
                if (thoughtMs != null) append(" · thought %.1f s".format(Locale.US, thoughtMs / 1000.0))
                if (termination == "cancelled") append(" · stopped")
                if (termination == "length") append(" · length limit")
                if (!json.isNull("steering")) append(" · steering %+.2f".format(Locale.US, steering.strength))
            }
            val stopped = termination == "cancelled"
            ChatStore.updateMessage(chat.id, answerId, true) { it.copy(text = answer, streaming = false, stats = stats, stopped = stopped,
                steering = json.optJSONObject("steering")?.toString(), reasoning = split?.reasoning, reasoningMs = thoughtMs) }
            update { s ->
                s.copy(phase = Phase.Ready, status = "Ready", lastResult = json.toString(2), lastSpeed = speed, generation = null)
            }
            if (!AppVisibility.visible) Notifications.answerFinished(app, chat.id, chat.title,
                answer.ifEmpty { "The model stopped while reasoning, before writing an answer." }, failed = false, stopped = stopped)
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
                        val body = try {
                            ChatStore.file(chat.id, attachment).readText()
                        } catch (error: Exception) {
                            throw AttachmentException("Cannot read attachment \"${attachment.name}\": ${error.message}")
                        }
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
