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

/** [Settings.context]: the largest window that fits while the model keeps its speed. */
const val CONTEXT_AUTO = 0
/** [Settings.context]: the model's trained length, or the largest window free memory allows. */
const val CONTEXT_MAX = -1
/** [Settings.maxTokens]: answers end only with the model or a full context window. */
const val ANSWER_UNLIMITED = 0

data class Settings(
    val systemPrompt: String = "You are Equity, a helpful assistant running privately on this phone. Answer in the user's language, clearly and concisely.",
    val maxTokens: Int = ANSWER_UNLIMITED,
    val temperature: Float = 0f,
    // The six big cores (one X4, five A720) on the Exynos 2400: 26% faster decoding than four, measured.
    val threads: Int = 6,
    /** A fixed window in tokens, [CONTEXT_AUTO] or [CONTEXT_MAX]; long messages and attachments still enlarge it. */
    val context: Int = CONTEXT_AUTO,
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
    val status: String = "",
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
    /** Context window of the loaded model, in tokens. */
    val contextSize: Int? = null,
) {
    val busy get() = steeringOperation || phase == Phase.Loading || phase == Phase.Generating || phase == Phase.Unloading
}

/** Owns the single native engine. Native calls run on one worker thread; only cancellation crosses threads. */
object EngineController {
    private lateinit var strings: android.content.res.Resources

    fun init(context: Context) {
        strings = context.applicationContext.resources
        flow.value = flow.value.copy(status = strings.getString(R.string.no_model_loaded))
    }

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
        /** 0 when unknown before loading (a picked file); the engine then caps the window at the trained length. */
        val trainedContext: Int = 0,
        val kvStorageMiBPer1k: Int = 0,
        val kvMemoryMiBPer1k: Int = 0,
    )
    private var loadedSource: ModelSource? = null
    private var trainedContext: Int? = null
    @Volatile private var cancelled = false

    // Images shown to the model per request: older images in a chat are described in text instead, so a long
    // conversation does not re-encode every photo on each turn.
    private const val MAX_CONTEXT_IMAGES = 4
    // Vision encoder working memory on top of the projector weights while images are encoded. With flash attention it
    // measured 35 MiB at 1024x1024 (Qwen3.6) and about 120 MiB for an 864x1536 photo at the 1,024-token cap
    // (Flash-Next, D038); without it the attention matrices alone take about 1 GiB at that size.
    private const val VISION_WORK_MIB = 256
    // Free memory left to the rest of the phone once a streamed model is loaded. The admission sum below matched
    // the measured peak RSS of a decode (3,088 MiB predicted 3,126 with a 512 MiB cache), so no further margin
    // is needed; Android also reclaims cached apps and swaps before it would stop the foreground app.
    private const val SYSTEM_RESERVE_MIB = 384
    // GPU mode on the Galaxy (929-token prompt, batch 2048): compute buffers 140 MiB above batch 512, and up to
    // 310 MiB of GPU staging while a prompt is read, released before the answer is written. Measured peak RSS.
    private const val GPU_WORK_MIB = 192
    private const val GPU_STAGING_MIB = 384
    // Room the prompt must leave for the answer; the context window is enlarged when it does not. Longer answers
    // continue until the window is full.
    private const val ANSWER_RESERVE = 2048
    // Windows chosen from free memory never start smaller than this, and stop here for files of unknown length.
    private const val MIN_AUTO_CONTEXT = 4096
    private const val MAX_UNKNOWN_CONTEXT = 131_072
    // With keys and values on storage, the page cache keeps the selected blocks attention reads (the engine's figure).
    private const val KV_PAGE_ALLOWANCE_MIB = 128
    // Storage left free when a window on storage is sized.
    private const val KV_STORAGE_MARGIN_MIB = 2048
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
                steeringNote = if (loadedSource == null) strings.getString(R.string.load_the_matching_model_to_use_its_vector) else strings.getString(R.string.verify_this_model_to_find_compatible_vectors)) }
            return
        }
        try {
            val profile = SteeringFiles.read(context, hash)
            val engine = flow.value
            require(profile == null || (profile.architecture == engine.architecture && profile.embedding == engine.embeddingSize &&
                profile.layer < (engine.decoderBlocks ?: 0))) { strings.getString(R.string.imported_steering_dimensions_do_not_match_this_model) }
            update { it.copy(modelSha256 = hash, steeringProfile = profile,
                steeringNote = if (profile == null) strings.getString(R.string.no_vector_imported_for_this_model) else null) }
        } catch (error: Exception) {
            update { it.copy(modelSha256 = hash, steeringProfile = null, steeringNote = strings.getString(R.string.steering_unavailable_1_s, error.message)) }
        }
    }

    /** Imports a vector for the model whose fingerprint is [modelHash] (null: whichever model the files name). */
    fun importSteering(context: Context, uris: List<Uri>, modelHash: String? = null) {
        val previous = flow.value
        if (previous.busy) return
        val app = context.applicationContext
        update { it.copy(phase = Phase.Loading, loadPercent = null, status = strings.getString(R.string.importing_steering_vector), error = null,
            steeringImportMessage = null, steeringMessageModel = modelHash, steeringOperation = true) }
        task {
            try {
                val imported = SteeringFiles.importVector(app, uris, modelHash) { cancelled }
                refreshSteering(app)
                val loaded = imported.modelHash == flow.value.modelSha256
                update { it.copy(steeringMessageModel = imported.modelHash,
                    steeringImportMessage = if (loaded) strings.getString(R.string.vector_imported_steering_is_available_in_this_model_s_chats)
                        else strings.getString(R.string.vector_imported_steering_appears_in_chats_while_this_model_is_loaded)) }
            } catch (error: Exception) {
                if (!cancelled) update { it.copy(error = error.message ?: strings.getString(R.string.steering_import_failed)) }
            } finally {
                update { it.copy(steeringRevision = it.steeringRevision + 1) }
                finishSteeringOperation(previous)
            }
        }
    }

    fun removeSteering(context: Context, modelHash: String) {
        val previous = flow.value
        if (previous.busy) return
        update { it.copy(phase = Phase.Loading, steeringOperation = true, status = strings.getString(R.string.removing_steering_vector),
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
        update { it.copy(phase = Phase.Loading, loadPercent = null, status = strings.getString(R.string.verifying_model_for_steering), error = null,
            steeringOperation = true) }
        task {
            try {
                val before = Os.stat(source.path)
                val digest = MessageDigest.getInstance("SHA-256")
                File(source.path).inputStream().use { input ->
                    val buffer = ByteArray(1024 * 1024)
                    var total = 0L
                    while (true) {
                        check(!cancelled) { strings.getString(R.string.model_verification_cancelled) }
                        val count = input.read(buffer)
                        if (count < 0) break
                        digest.update(buffer, 0, count)
                        total += count
                    }
                    require(total == source.fileBytes) { strings.getString(R.string.model_size_changed) }
                }
                val after = Os.stat(source.path)
                require(before.st_size == after.st_size && before.st_mtime == after.st_mtime) { strings.getString(R.string.model_changed_during_verification) }
                val hash = digest.digest().joinToString("") { "%02x".format(it) }
                // Reload shared/resident tensors too, so the fingerprint describes the weights actually in use.
                guardLoad {
                    loadSource(context.applicationContext, source.copy(hash = hash))
                    val loaded = Os.stat(source.path)
                    require(loaded.st_size == before.st_size && loaded.st_mtime == before.st_mtime) {
                        strings.getString(R.string.model_changed_while_reloading_verified_weights)
                    }
                }
            } catch (error: Exception) {
                if (!cancelled) update { it.copy(error = error.message ?: strings.getString(R.string.model_verification_failed)) }
            } finally {
                update { it.copy(phase = if (loadedSource == null) Phase.Empty else Phase.Ready, loadPercent = null,
                    status = if (loadedSource == null) strings.getString(R.string.no_model_loaded) else strings.getString(R.string.ready), steeringOperation = false) }
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
                require(ModelFiles.installed(app, model.artifact)) { strings.getString(R.string.value_1_s_is_not_downloaded, model.name) }
                val projector = model.projector?.takeIf { ModelFiles.installed(app, it) }
                val visionMiB = projector?.let { (it.sizeBytes shr 20).toInt() + VISION_WORK_MIB } ?: 0
                loadSource(app, ModelSource(path.absolutePath, model.sha256, model.streamed, model.residentMiB,
                    model.contextMiBPer1k, path.length(), settings,
                    projector?.let { ModelFiles.final(app, it).absolutePath }, visionMiB, model.stream, model.trainedContext,
                    model.kvStorageMiBPer1k, model.kvMemoryMiBPer1k))
            }
        }
    }

    /** Loads a user-selected local GGUF through a seekable descriptor, without copying it. */
    fun load(context: Context, uri: Uri) {
        val app = context.applicationContext
        val settings = SettingsStore.state.value
        startLoad(app, null, strings.getString(R.string.local_model))
        task {
            guardLoad {
                NativeBridge.unload()
                closeDescriptor()
                val pfd = requireNotNull(app.contentResolver.openFileDescriptor(uri, "r")) { strings.getString(R.string.cannot_open_the_file) }
                descriptor = pfd
                val stat = Os.fstat(pfd.fileDescriptor)
                require(OsConstants.S_ISREG(stat.st_mode) && stat.st_size > 0) {
                    strings.getString(R.string.choose_a_local_seekable_file_cloud_providers_and_pipes_are_unsupported)
                }
                val magic = ByteArray(4)
                require(Os.pread(pfd.fileDescriptor, magic, 0, 4, 0) == 4 && magic.decodeToString() == "GGUF") {
                    strings.getString(R.string.the_selected_file_is_not_a_gguf_model)
                }
                val sizeMiB = (stat.st_size shr 20).toInt()
                val streamed = sizeMiB + 1536 > availableMiB(app)
                // Small files are hashed for provenance; large streamed artifacts are not re-read on every load.
                var hash: String? = null
                if (!streamed) {
                    update { it.copy(status = strings.getString(R.string.checking_file_integrity)) }
                    val digest = MessageDigest.getInstance("SHA-256")
                    val buffer = ByteArray(1 shl 20)
                    var offset = 0L
                    while (offset < stat.st_size) {
                        check(!cancelled) { strings.getString(R.string.loading_cancelled) }
                        val count = Os.pread(pfd.fileDescriptor, buffer, 0, minOf(buffer.size.toLong(), stat.st_size - offset).toInt(), offset)
                        require(count > 0) { strings.getString(R.string.incomplete_model_read) }
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
                    settings, stream = sized?.stream ?: StreamProfile(), trainedContext = sized?.trainedContext ?: 0,
                    kvStorageMiBPer1k = sized?.kvStorageMiBPer1k ?: 0, kvMemoryMiBPer1k = sized?.kvMemoryMiBPer1k ?: 0))
            }
        }
    }

    private fun startLoad(context: Context, id: String?, name: String) {
        update {
            it.copy(phase = Phase.Loading, modelId = id, modelName = name, loadPercent = null, error = null,
                status = strings.getString(R.string.preparing_1_s, name), configuration = null, vision = false, lastModelId = id ?: it.lastModelId,
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
                EngineState(status = strings.getString(R.string.no_model_loaded), lastModelId = it.lastModelId,
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
        val transientMiB = prefillMiB + gpuStagingMiB
        val floor = maxOf(512, layerMiB)
        val cache = settings.expertCacheMiB.takeIf { it > 0 }
            ?: available?.let { (it - residentMiB - workMiB - transientMiB - SYSTEM_RESERVE_MIB).coerceIn(floor, 3072) }
            ?: floor
        val shortfallMiB = maxOf(0, layerMiB - cache)
        // Images are encoded with the expert cache emptied (images_release_expert_cache) and the projector is released
        // before the prompt is read, so the encoder's memory replaces the cache instead of adding to it.
        val imagePhaseMiB = residentMiB + visionMiB + gpuStagingMiB + workMiB + SYSTEM_RESERVE_MIB
        return StreamPlan(batch, prefillMiB, cache, shortfallMiB,
            maxOf(residentMiB + cache + shortfallMiB + transientMiB + workMiB + SYSTEM_RESERVE_MIB, imagePhaseMiB))
    }

    /** Free memory, in MiB, that [model] needs to load with full prompt batches and the next-layer allowance. */
    fun fullSpeedMemoryMiB(model: CatalogModel): Int {
        val settings = SettingsStore.state.value
        // A window chosen from free memory starts at its smallest size.
        val tokens = settings.context.takeIf { it > 0 } ?: MIN_AUTO_CONTEXT
        val kvMiB = if (model.kvStorageMiBPer1k > 0) model.kvMemoryMiBPer1k * tokens / 1024 + KV_PAGE_ALLOWANCE_MIB
            else model.contextMiBPer1k * tokens / 1024
        if (!model.streamed) return (model.sizeBytes shr 20).toInt() + kvMiB + 512 + 512
        return streamPlan(null, model.residentMiB, kvMiB, settings, 0, model.stream,
            if (settings.gpu) 2048 else 512, 512).requiredMiB
    }

    fun availableMiB(context: Context): Int {
        val memory = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        return (memory.availMem shr 20).toInt()
    }

    /**
     * The context window to load. Fixed sizes are kept. Otherwise the KV cache gets memory nothing else needs: Auto
     * takes half of what is left once a streamed model has its full-speed expert cache (resident models: once the
     * model fits), leaving the rest to the phone; Max takes all of it with the expert cache at its floor, up to the
     * trained length. Long messages and attachments still enlarge the window afterwards.
     */
    private fun resolveContext(available: Int, streamed: Boolean, residentMiB: Int, contextMiBPer1k: Int, fileBytes: Long,
                               settings: Settings, visionMiB: Int, stream: StreamProfile, trained: Int,
                               extraMiB: Int = 0, storageTokens: Int = Int.MAX_VALUE): Int {
        if (settings.context > 0) return if (trained > 0) minOf(settings.context, trained) else settings.context
        val max = settings.context == CONTEXT_MAX
        val gpuStagingMiB = if (settings.gpu) GPU_STAGING_MIB else 0
        val gpuWorkMiB = if (settings.gpu) GPU_WORK_MIB else 0
        val fixedMiB = if (!streamed) {
            (fileBytes shr 20).toInt() + 512 + gpuWorkMiB + 512 + maxOf(visionMiB, gpuStagingMiB)
        } else {
            val batch = if (settings.gpu) 2048 else 512
            val workMiB = if (batch == 2048) 512 + GPU_WORK_MIB else stream.computeMiBPer512
            val layerMiB = stream.layerWorkingSetMiB(batch)
            val floor = maxOf(512, layerMiB)
            val cache = settings.expertCacheMiB.takeIf { it > 0 } ?: if (max) floor else maxOf(floor, stream.speedCacheMiB)
            // The image encoder takes the emptied expert cache's place (streamPlan), not memory beside it.
            maxOf(residentMiB + cache + maxOf(0, layerMiB - cache) + gpuStagingMiB + workMiB + SYSTEM_RESERVE_MIB,
                residentMiB + visionMiB + gpuStagingMiB + workMiB + SYSTEM_RESERVE_MIB)
        }
        val spareMiB = maxOf(0, available - fixedMiB - extraMiB).toLong().let { if (max) it else it / 2 }
        val tokens = (spareMiB * 1024 / maxOf(1, contextMiBPer1k) / 1024 * 1024).toInt()
        val limit = minOf(if (trained > 0) trained else MAX_UNKNOWN_CONTEXT, maxOf(MIN_AUTO_CONTEXT, storageTokens / 1024 * 1024))
        return tokens.coerceIn(minOf(MIN_AUTO_CONTEXT, limit), limit)
    }

    /** [visionMiB] is memory kept free for image encoding; 0 for a text-only load. */
    private fun loadOptions(context: Context, streamed: Boolean, residentMiB: Int, contextMiBPer1k: Int, fileBytes: Long,
                            settings: Settings, visionMiB: Int, stream: StreamProfile, trained: Int,
                            kvStorageMiBPer1k: Int = 0, kvMemoryMiBPer1k: Int = 0): JSONObject {
        val available = availableMiB(context)
        // Keys and values on storage: memory holds the indexer rows and the pages attention reads; the window is also
        // bounded by free storage (the engine reserves its whole file at load).
        val onStorage = kvStorageMiBPer1k > 0
        val memoryPerK = if (onStorage) kvMemoryMiBPer1k else contextMiBPer1k
        val allowanceMiB = if (onStorage) KV_PAGE_ALLOWANCE_MIB else 0
        val storageTokens = if (!onStorage) Int.MAX_VALUE else {
            val freeMiB = ((ModelFiles.freeBytes(context) shr 20) - KV_STORAGE_MARGIN_MIB).coerceAtLeast(0)
            (freeMiB * 1024 / kvStorageMiBPer1k).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        }
        val contextSize = resolveContext(available, streamed, residentMiB, memoryPerK, fileBytes, settings, visionMiB,
            stream, trained, allowanceMiB, storageTokens)
        // Context state and compute buffers are allocated at load; admitting without them invites the low-memory killer.
        // Prompt batches of 512 tokens take the tiled GEMM path and need larger compute buffers.
        // GPU mode reads prompts in batches of 2048 tokens (fewer passes over the streamed experts, fuller GPU
        // tiles): larger compute buffers, plus staging for one GPU operation that is released after each prompt.
        val gpuWorkMiB = if (settings.gpu) GPU_WORK_MIB else 0
        val gpuStagingMiB = if (settings.gpu) GPU_STAGING_MIB else 0
        val kvMiB = memoryPerK * contextSize / 1024 + allowanceMiB
        val workMiB = kvMiB + 512 + gpuWorkMiB
        val firstBatch = if (settings.gpu) 2048 else 512
        val options = JSONObject().put("context", contextSize).put("batch", firstBatch)
            .put("threads", settings.threads)
        // A picked file's trained length is known only to the engine, which caps an automatic window at it.
        if (settings.context <= 0 && trained <= 0) options.put("context_up_to_trained", true)
        if (onStorage) options.put("kv_file_dir", File(context.filesDir, "kv").apply { mkdirs() }.path)
        if (settings.gpu) options.put("gpu", true)
        if (!streamed) {
            val fileMiB = (fileBytes shr 20).toInt()
            val requiredMiB = fileMiB + workMiB + 512 + maxOf(visionMiB, gpuStagingMiB)
            require(requiredMiB <= available) {
                strings.getString(R.string.this_model_needs_about_1_d_mib_of_free_memory_and_2_d_mib_are_available_clo, requiredMiB, available)
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
        throw IllegalArgumentException(strings.getString(R.string.this_model_needs_about_1_d_mib_of_free_memory_and_2_d_mib_are_available_clo_299, required, available))
    }

    private fun loadSource(context: Context, source: ModelSource, growing: Boolean = false) {
        // Reclaim the previous context and expert cache before sizing their replacements from free memory.
        NativeBridge.unload()
        loadedSource = null
        fun options(visionMiB: Int) = loadOptions(context, source.streamed, source.residentMiB,
            source.contextMiBPer1k, source.fileBytes, source.settings, visionMiB, source.stream, source.trainedContext,
            source.kvStorageMiBPer1k, source.kvMemoryMiBPer1k)
        // A streamed model empties its expert cache before encoding images, which the admission above counts on.
        fun withProjector(path: String) = options(source.visionMiB).put("mmproj", path)
            .apply { if (source.streamed) put("images_release_expert_cache", true) }
        val withVision = source.projectorPath?.let { path ->
            if (growing) withProjector(path) else runCatching { withProjector(path) }.getOrNull()
        }
        val options = withVision ?: options(0)
        source.hash?.let { options.put("model_sha256", it) }
        // An expert pack beside the model (eqt-pack, D032) holds its experts one contiguous read each; after
        // `eqt-pack move` the model file has no experts of its own and the engine refuses it without the pack.
        if (source.streamed) {
            File(source.path.removeSuffix(".gguf") + ".eqtpack").takeIf { it.isFile }?.let { options.put("expert_pack", it.path) }
        }
        val note = if (source.projectorPath != null && withVision == null) strings.getString(R.string.images_off_not_enough_free_memory) else null
        loadNative(source.path, options, source.hash, note, growing)
        loadedSource = source.copy(settings = source.settings.copy(gpu = options.optBoolean("gpu")),
            projectorPath = source.projectorPath.takeIf { withVision != null })
        refreshSteering(context)
    }

    private fun loadNative(path: String, options: JSONObject, hash: String?, note: String?, growing: Boolean) {
        val progress = ByteSink { bytes ->
            val percent = bytes.decodeToString().toIntOrNull()
            update { it.copy(loadPercent = percent, status = strings.getString(R.string.loading_1_d, percent ?: 0)) }
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
                    gpuNote = strings.getString(R.string.gpu_unavailable)
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
            append(strings.getString(R.string.value_1_d_threads_2_d_context, info.getInt("threads"), info.getInt("context")))
            if (info.getInt("batch") < 512) append(strings.getString(R.string.value_1_d_token_prompt_batches, info.getInt("batch")))
            if (streaming != null) {
                append(strings.getString(R.string.value_1_d_mib_expert_cache, streaming.getLong("effective_budget_bytes") shr 20))
                if (!options.optBoolean("direct_io")) append(strings.getString(R.string.buffered_reads))
            }
            if (vision) append(" · images")
            note?.let { append(" · ").append(it) }
            gpuNote?.let { append(" · ").append(it) }
        }
        update {
            it.copy(phase = if (growing) Phase.Generating else Phase.Ready, loadPercent = null,
                lastResult = null, lastSpeed = null, configuration = configuration,
                status = if (growing) strings.getString(R.string.reading_your_message) else strings.getString(R.string.ready), vision = vision)
                .copy(modelSha256 = hash, architecture = info.getString("architecture"),
                    embeddingSize = info.getInt("embedding_size"), decoderBlocks = info.getInt("decoder_blocks"),
                    thinkingSupported = info.optBoolean("thinking_supported"), contextSize = info.getInt("context"))
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
                strings.getString(R.string.this_message_and_its_answer_need_1_d_tokens_exceeding_the_model_s_2_d_token, required, limit)
            }
            val expanded = (((required + 2047) / 2048) * 2048).coerceAtMost(limit)
            check(!cancelled) { strings.getString(R.string.generation_cancelled) }
            update { it.copy(status = strings.getString(R.string.expanding_context_to_1_d_tokens, expanded)) }
            try {
                loadSource(context, source.copy(settings = source.settings.copy(context = expanded)), growing = true)
            } catch (error: Exception) {
                runCatching { NativeBridge.unload() }
                closeDescriptor()
                throw IllegalStateException(strings.getString(R.string.this_message_needs_1_d_context_tokens_could_not_allocate_a_larger_context_2, required, error.message), error)
            }
            check(!cancelled) { strings.getString(R.string.generation_cancelled) }
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
        update { it.copy(phase = Phase.Unloading, status = strings.getString(R.string.unloading)) }
        task(prepare = false) {
            NativeBridge.unload()
            closeDescriptor()
            update { EngineState(status = strings.getString(R.string.model_unloaded), lastModelId = it.lastModelId) }
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
            update { EngineState(status = strings.getString(R.string.no_model_loaded), lastModelId = it.lastModelId, error = message) }
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
            it.copy(phase = Phase.Generating, status = strings.getString(R.string.reading_your_message), error = null,
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
                // Without a limit the answer may run to the end of the model's trained context, as far as the
                // loaded window allows; the prompt only has to leave ANSWER_RESERVE tokens of room.
                val maxTokens = settings.maxTokens.takeIf { it > 0 } ?: trainedContext ?: (1 shl 20)
                request.put("messages", messages).put("max_tokens", maxTokens)
                    .put("output_reserve", minOf(maxTokens, ANSWER_RESERVE))
                    .put("temperature", settings.temperature.toDouble()).put("seed", 42).put("thinking", thinking)
                    .put("ignore_eos", false)
                // A chat that keeps its state restores it when another chat or a reload replaced it in memory, and
                // saves it after each answer. It needs a verified model, and thinking turns are rebuilt anyway
                // (their reasoning is not sent back), so they neither read nor write it.
                if (chat.session && modelHash != null && !thinking) {
                    request.put("session", JSONObject().put("path", ChatStore.sessionFile(chat.id).absolutePath))
                }
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
                                !split.closed -> strings.getString(R.string.thinking)
                                visible -> strings.getString(R.string.writing)
                                else -> strings.getString(R.string.generating_tokens_no_visible_text_yet)
                            }, generation = s.generation?.copy(tokens = tokens, hasVisibleText = visible, thinking = !split.closed)) }
                    }
                })
            } catch (error: Exception) {
                val failure = error.message ?: error.javaClass.simpleName
                val partial = decodeComplete(output.toByteArray())
                val split = if (thinking) splitThinking(partial) else ThinkingSplit("", partial, true)
                ChatStore.updateMessage(chat.id, answerId, true) {
                    it.copy(streaming = false, failed = true, text = split.answer.ifEmpty { strings.getString(R.string.generation_failed_n_n_1_s, failure) },
                        reasoning = split.reasoning.takeIf { thinking && it.isNotEmpty() }, reasoningMs = reasoningMs,
                        stats = failure.takeIf { output.size() > 0 })
                }
                update { s ->
                    if (loadedSource == null) EngineState(lastModelId = s.lastModelId, error = failure)
                    else s.copy(phase = Phase.Ready, status = strings.getString(R.string.ready), error = failure, generation = null)
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
            val answer = split?.answer ?: json.getString("text").ifEmpty { strings.getString(R.string.no_answer) }
            val thoughtMs = if (thinking) json.getDouble("reasoning_ms").toLong() else null
            val termination = json.getString("termination")
            val speed = json.getDouble("decode_tokens_per_second")
            val stats = buildString {
                append("%.1f tok/s".format(Locale.US, speed))
                append(" · ${json.getInt("generated_tokens")} tokens")
                if (!json.isNull("ttft_ms")) append(strings.getString(R.string.first_token_1_1f_s, json.getDouble("ttft_ms") / 1000))
                if (thoughtMs != null) append(strings.getString(R.string.thought_1_1f_s, thoughtMs / 1000.0))
                if (termination == "cancelled") append(" · stopped")
                if (termination == "length") append(strings.getString(R.string.length_limit))
                if (termination == "context") append(strings.getString(R.string.context_full))
                json.optJSONObject("session")?.takeIf { it.optString("status") == "restored" }?.let {
                    append(strings.getString(R.string.resumed_1_d_saved_tokens, it.optInt("restored_tokens")))
                }
                if (!json.isNull("steering")) append(strings.getString(R.string.steering_1_2f, steering.strength))
            }
            val stopped = termination == "cancelled"
            val contextTokens = json.getInt("prompt_tokens") + json.getInt("generated_tokens")
            ChatStore.updateMessage(chat.id, answerId, true) { it.copy(text = answer, streaming = false, stats = stats, stopped = stopped,
                steering = json.optJSONObject("steering")?.toString(), reasoning = split?.reasoning, reasoningMs = thoughtMs,
                contextTokens = contextTokens) }
            update { s ->
                s.copy(phase = Phase.Ready, status = strings.getString(R.string.ready), lastResult = json.toString(2), lastSpeed = speed, generation = null)
            }
            if (!AppVisibility.visible) Notifications.answerFinished(app, chat.id, chat.title,
                answer.ifEmpty { strings.getString(R.string.the_model_stopped_while_reasoning_before_writing_an_answer) }, failed = false, stopped = stopped)
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
                            throw AttachmentException(strings.getString(R.string.cannot_read_attachment_1_s_2_s, attachment.name, error.message))
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
