package org.equity.app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.StatFs
import androidx.core.app.NotificationCompat
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/** One shard of a model the repository splits with gguf-split. */
data class ArtifactPart(val file: String, val sizeBytes: Long)

/**
 * A pinned, hash-verified file the app can download: a model or a vision projector. A file the repository ships
 * as gguf-split shards ([parts]) is downloaded as their merge, written in place while the shards stream: its
 * merged header must match [headerSha256] before any weights are written, and the whole file [sha256].
 */
data class Artifact(
    val id: String,
    val name: String,
    val repository: String,
    val revision: String,
    val file: String,
    val sizeBytes: Long,
    val sha256: String,
    /** Large files wait for an unmetered network instead of consuming mobile data. */
    val unmeteredOnly: Boolean,
    val parts: List<ArtifactPart> = emptyList(),
    val headerBytes: Long = 0,
    val headerSha256: String? = null,
) {
    val url get() = url(file)
    fun url(path: String) = "https://huggingface.co/$repository/resolve/$revision/$path"
}

/**
 * Memory figures of a streamed model. A prompt batch of b tokens keeps the share 1 - (1 - k/E)^b of a layer's
 * routed experts resident at once (top-k of E experts) and needs compute buffers linear in b.
 */
data class StreamProfile(
    /** Routed-expert MiB of the largest layer. */
    val layerExpertsMiB: Int = 0,
    val experts: Int = 0,
    val expertsUsed: Int = 0,
    /** Compute buffers of a 512-token batch, in MiB. */
    val computeMiBPer512: Int = 512,
    /** Engine load options measured for this model, applied over the streamed defaults. */
    val tuning: Map<String, Any> = emptyMap(),
    /**
     * Expert cache the model needs for its measured speed; the automatic context window takes only memory beyond it
     * (Flash-Next: 1,792 MiB in D032; the 35B builds: 2 GiB, D024).
     */
    val speedCacheMiB: Int = 2048,
) {
    /** MiB of one layer's experts that a batch of [batch] tokens holds resident. */
    fun layerWorkingSetMiB(batch: Int): Int {
        if (experts <= 0 || expertsUsed <= 0) return layerExpertsMiB
        val share = 1 - Math.pow(1 - expertsUsed.toDouble() / experts, batch.toDouble())
        return Math.ceil(layerExpertsMiB * share).toInt()
    }
}

/** A pinned, hash-verified model the app can download and run. */
data class CatalogModel(
    val id: String,
    val name: String,
    @androidx.annotation.StringRes val taglineRes: Int,
    @androidx.annotation.StringRes val descriptionRes: Int,
    val repository: String,
    val revision: String,
    val file: String,
    val sizeBytes: Long,
    val sha256: String,
    val license: String,
    /** Routed experts stream from storage; only shared weights stay resident. */
    val streamed: Boolean,
    /**
     * Approximate resident weight footprint in MiB, used for RAM admission. For streamed models it excludes the
     * routed experts and the token embedding table, which the engine maps from the file.
     */
    val residentMiB: Int,
    /** KV cache plus recurrent state per 1024 context tokens, in MiB (F16 cache). */
    val contextMiBPer1k: Int,
    /** The context length the model was trained for (GGUF context_length). */
    val trainedContext: Int,
    /**
     * Sparse-attention models whose attention keys and values can live on storage (kv_file_dir, D034): storage and
     * memory per 1024 context tokens then; 0 keeps them in memory.
     */
    val kvStorageMiBPer1k: Int = 0,
    val kvMemoryMiBPer1k: Int = 0,
    /** Vision encoder (mmproj) that lets this model read images; null for text-only models. */
    val projector: Artifact? = null,
    /** Parameter count shown on the model's badge. */
    val badge: String = "",
    val stream: StreamProfile = StreamProfile(),
    val parts: List<ArtifactPart> = emptyList(),
    val headerBytes: Long = 0,
    val headerSha256: String? = null,
) {
    val artifact get() = Artifact(id, name, repository, revision, file, sizeBytes, sha256, unmeteredOnly = streamed,
        parts = parts, headerBytes = headerBytes, headerSha256 = headerSha256)
}

object Catalog {
    // Unsloth's F16 conversion of the Qwen3.6 vision encoder, shared by both 35B builds (same repository and
    // revision as the language weights).
    private val qwen36Projector = Artifact(
        id = "qwen3.6-35b-a3b-mmproj-f16", name = "Qwen3.6 image support",
        repository = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", revision = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        file = "mmproj-F16.gguf", sizeBytes = 899_283_584L,
        sha256 = "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160",
        unmeteredOnly = true,
    )

    // Unsloth's F16 conversion of the Qwen3.8 Flash-Next vision encoder (qwen3vl_merger, 449M parameters), from the
    // same repository and revision as the language weights.
    private val flashNextProjector = Artifact(
        id = "qwen3.8-flash-next-mmproj-f16", name = "Qwen3.8 Flash Next image support",
        repository = "unsloth/Qwen3.8-Flash-Next-GGUF", revision = "766911a6b7369840a91dbcd95f9f997acaab6cd6",
        file = "mmproj-F16.gguf", sizeBytes = 904_004_000L,
        sha256 = "1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980",
        unmeteredOnly = true,
    )

    // Unsloth's F16 conversion of the Qwen3.5 9B vision encoder (qwen3vl_merger, 456M parameters, projecting to 4096),
    // shared by both 9B builds (same repository and revision as the language weights).
    private val qwen35_9bProjector = Artifact(
        id = "qwen3.5-9b-mmproj-f16", name = "Qwen3.5 9B image support",
        repository = "unsloth/Qwen3.5-9B-GGUF", revision = "3885219b6810b007914f3a7950a8d1b469d598a5",
        file = "mmproj-F16.gguf", sizeBytes = 918_166_080L,
        sha256 = "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f",
        unmeteredOnly = true,
    )

    val models = listOf(
        CatalogModel(
            id = "qwen3-0.6b-q8", name = "Qwen3 0.6B", taglineRes = R.string.model_tagline_qwen3_0_6b,
            descriptionRes = R.string.model_description_qwen3_0_6b,
            repository = "Qwen/Qwen3-0.6B-GGUF", revision = "23749fefcc72300e3a2ad315e1317431b06b590a",
            file = "Qwen3-0.6B-Q8_0.gguf", sizeBytes = 639_446_688L,
            sha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
            license = "Apache-2.0", streamed = false, residentMiB = 610, contextMiBPer1k = 112, trainedContext = 40_960,
            badge = "0.6B",
        ),
        // Qwen3.5 9B (qwen35, dense, Gated DeltaNet with full attention every fourth of 32 layers): 8 attention layers
        // x 4 KV heads x 256 x K and V in F16 = 32 KiB per token, plus about 48 MiB of recurrent state. Resident, so
        // admission counts the whole file; the 3-bit build is for phones with less free memory.
        CatalogModel(
            id = "qwen3.5-9b-q4km", name = "Qwen3.5 9B · 4-bit",
            taglineRes = R.string.model_tagline_qwen35_9b_q4, descriptionRes = R.string.model_description_qwen35_9b_q4,
            repository = "unsloth/Qwen3.5-9B-GGUF", revision = "3885219b6810b007914f3a7950a8d1b469d598a5",
            file = "Qwen3.5-9B-Q4_K_M.gguf", sizeBytes = 5_680_522_464L,
            sha256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
            license = "Apache-2.0", streamed = false, residentMiB = 5420, contextMiBPer1k = 48, trainedContext = 262_144,
            projector = qwen35_9bProjector, badge = "9B",
        ),
        CatalogModel(
            id = "qwen3.5-9b-ud-iq3xxs", name = "Qwen3.5 9B · 3-bit",
            taglineRes = R.string.model_tagline_qwen35_9b_iq3, descriptionRes = R.string.model_description_qwen35_9b_iq3,
            repository = "unsloth/Qwen3.5-9B-GGUF", revision = "3885219b6810b007914f3a7950a8d1b469d598a5",
            file = "Qwen3.5-9B-UD-IQ3_XXS.gguf", sizeBytes = 4_016_235_744L,
            sha256 = "40d0f32cd3030b04f0784139a589fb63e876cfbf8667d56311b79783c74fd149",
            license = "Apache-2.0", streamed = false, residentMiB = 3830, contextMiBPer1k = 48, trainedContext = 262_144,
            projector = qwen35_9bProjector, badge = "9B",
        ),
        CatalogModel(
            id = "qwen3.6-35b-a3b-iq2m", name = "Qwen3.6 35B-A3B · 2.7-bit",
            taglineRes = R.string.model_tagline_qwen36_35b_iq2m,
            descriptionRes = R.string.model_description_qwen36_35b_iq2m,
            repository = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", revision = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
            file = "Qwen3.6-35B-A3B-UD-IQ2_M.gguf", sizeBytes = 11_882_969_376L,
            sha256 = "b989da14b63b29bd469cb1fab5bd16b422ab921787569b4b2bb4fd61e26129b8",
            license = "Apache-2.0", streamed = true, residentMiB = 1430, contextMiBPer1k = 40, trainedContext = 262_144,
            projector = qwen36Projector, badge = "35B",
            stream = StreamProfile(layerExpertsMiB = 240, experts = 256, expertsUsed = 8),
        ),
        CatalogModel(
            id = "qwen3.6-35b-a3b-q4km", name = "Qwen3.6 35B-A3B · 4-bit",
            taglineRes = R.string.model_tagline_qwen36_35b_q4km,
            descriptionRes = R.string.model_description_qwen36_35b_q4km,
            repository = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", revision = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
            file = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", sizeBytes = 22_663_387_424L,
            sha256 = "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b",
            license = "Apache-2.0", streamed = true, residentMiB = 1935, contextMiBPer1k = 40, trainedContext = 262_144,
            projector = qwen36Projector, badge = "35B",
            stream = StreamProfile(layerExpertsMiB = 466, experts = 256, expertsUsed = 8),
        ),
        // Unsloth splits the 78.9 GB build into three gguf-split shards (configs/flash-next-model.json); the app
        // writes their merge, byte-identical to llama-gguf-split --merge of the same shards.
        CatalogModel(
            id = "qwen3.8-flash-next-ud-q2kxl", name = "Qwen3.8 Flash Next · Q2_K_XL",
            taglineRes = R.string.model_tagline_qwen38_flash_next,
            descriptionRes = R.string.model_description_qwen38_flash_next,
            repository = "unsloth/Qwen3.8-Flash-Next-GGUF", revision = "766911a6b7369840a91dbcd95f9f997acaab6cd6",
            file = "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf", sizeBytes = 78_869_128_640L,
            sha256 = "f26a5e679710fb4cd048f481ff082d9af9c79fbd640d1867ef58c425518659d7",
            // Shared weights 3,372 MiB, recurrent state 113 MiB; the expert banks and n-gram table are file-backed.
            license = "Qwen Community 1.0", streamed = true, residentMiB = 3500, contextMiBPer1k = 30, trainedContext = 262_144,
            // QSA reads 2,048 selected cells per layer: its 24 KiB per token of keys and values stay on storage and
            // only the 6 KiB of indexer rows in memory.
            kvStorageMiBPer1k = 24, kvMemoryMiBPer1k = 6,
            projector = flashNextProjector, badge = "125B",
            // Compute buffers measured on the host: 518 MiB at 512 tokens and 4,096 context, 534 at 16,384; linear
            // in the batch. 512 experts: a prompt shorter than ~118 tokens uses under 90% of a layer's experts, so
            // reading the next layer whole wastes reads (52 tokens: 54 GB read instead of 31, measured on the host).
            // Lookahead reads in a pool beside the cache, two layers ahead (D029, Galaxy): 1.47 -> 1.68 tokens/s
            // at 1 GiB, 1.92 at 1.75 GiB, identical tokens. The pool holds about 60 MiB within the reserve.
            // Demand reads first and at most 3 speculative reads in flight (D030, Galaxy, 1.75 GiB): 2.12 -> 2.39.
            // With an expert pack beside the model (D032): 2 whole-expert reads in flight, demanded experts in two
            // parts, and the lookahead router as Q8_0 with the four-row kernel: 2.45 -> 2.89-3.06, same tokens.
            stream = StreamProfile(layerExpertsMiB = 916, experts = 512, expertsUsed = 10, computeMiBPer512 = 540,
                tuning = mapOf("prefill_prefetch_min_coverage" to 0.9, "speculative_pool" to true,
                    "lookahead_layers" to 2, "lookahead2_count" to 5, "read_priority" to true, "prefetch_inflight" to 2,
                    "expert_pack_split" to 2, "lookahead_router_q8" to true), speedCacheMiB = 1792),
            parts = listOf(
                ArtifactPart("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", 10_946_624L),
                ArtifactPart("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf", 49_979_779_296L),
                ArtifactPart("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf", 28_878_402_944L),
            ),
            headerBytes = 11_024_320L, headerSha256 = "2106fcb2ff86c4addea4ecec7895300bbac4ef28c5d42db4742b3181a56ca31a",
        ),
    )

    fun byId(id: String?) = models.firstOrNull { it.id == id }

    /** Every downloadable file, models first; the index gives each its own download notification. */
    val artifacts: List<Artifact> = (models.map { it.artifact } + models.mapNotNull { it.projector }).distinctBy { it.id }

    fun artifact(id: String?) = artifacts.firstOrNull { it.id == id }
}

object ModelFiles {
    fun dir(context: Context, artifact: Artifact) = File(context.filesDir, "models/${artifact.id}")
    fun final(context: Context, artifact: Artifact) = File(dir(context, artifact), artifact.file)
    fun part(context: Context, artifact: Artifact) = File(dir(context, artifact), artifact.file + ".part")
    private fun marker(context: Context, artifact: Artifact) = File(dir(context, artifact), artifact.file + ".sha256")

    /** True only after the full download matched the pinned SHA-256. */
    fun installed(context: Context, artifact: Artifact): Boolean {
        val file = final(context, artifact)
        val marker = marker(context, artifact)
        return file.isFile && file.length() == artifact.sizeBytes && marker.isFile && marker.readText().trim() == artifact.sha256
    }

    fun partialBytes(context: Context, artifact: Artifact) = part(context, artifact).takeIf { it.isFile }?.length() ?: 0L

    fun markVerified(context: Context, artifact: Artifact) = marker(context, artifact).writeText(artifact.sha256)

    fun delete(context: Context, artifact: Artifact) = dir(context, artifact).deleteRecursively()

    fun freeBytes(context: Context) = StatFs(context.filesDir.path).availableBytes
}

object Downloads {
    const val KEY_ID = "id"
    const val KEY_DONE = "done"
    const val KEY_TOTAL = "total"
    const val KEY_SPEED = "speed"
    const val KEY_PHASE = "phase"
    const val KEY_ERROR = "error"
    const val PHASE_VERIFYING = "verifying"
    const val PHASE_DOWNLOADING = "downloading"

    private fun name(artifact: Artifact) = "download-${artifact.id}"

    fun start(context: Context, artifact: Artifact) {
        // Large files wait for an unmetered network instead of consuming mobile data.
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(if (artifact.unmeteredOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
            .setRequiresStorageNotLow(true)
            .build()
        val request = OneTimeWorkRequestBuilder<DownloadWorker>()
            .setInputData(workDataOf(KEY_ID to artifact.id))
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .addTag("model-download")
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(name(artifact), ExistingWorkPolicy.REPLACE, request)
    }

    /** Stops the transfer and keeps the partial file for a later resume. */
    fun pause(context: Context, artifact: Artifact) {
        WorkManager.getInstance(context).cancelUniqueWork(name(artifact))
    }

    /** Cancels any transfer, waits for the worker to release the file, then removes all local data. */
    fun delete(context: Context, artifact: Artifact, done: () -> Unit) {
        val app = context.applicationContext
        val manager = WorkManager.getInstance(app)
        manager.cancelUniqueWork(name(artifact))
        Thread {
            runCatching {
                while (manager.getWorkInfosForUniqueWork(name(artifact)).get().any { !it.state.isFinished }) Thread.sleep(100)
            }
            ModelFiles.delete(app, artifact)
            done()
        }.start()
    }

    fun observe(context: Context, artifact: Artifact): Flow<WorkInfo?> =
        WorkManager.getInstance(context).getWorkInfosForUniqueWorkFlow(name(artifact)).map { it.firstOrNull() }
}

/** Resumable, hash-verified download of one catalog file, run as a data-sync foreground job. */
class DownloadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    private class IntegrityException(message: String) : Exception(message)

    override suspend fun doWork(): Result {
        val artifact = Catalog.artifact(inputData.getString(Downloads.KEY_ID))
            ?: return Result.failure(workDataOf(Downloads.KEY_ERROR to applicationContext.getString(R.string.unknown_file)))
        if (ModelFiles.installed(applicationContext, artifact)) return Result.success()
        // A foreground start can be refused when a retry runs in the background; the transfer still proceeds.
        runCatching { setForeground(foregroundInfo(artifact, 0, artifact.sizeBytes, applicationContext.getString(R.string.starting))) }
        return withContext(Dispatchers.IO) {
            try {
                download(artifact)
                Result.success()
            } catch (error: CancellationException) {
                throw error
            } catch (error: IntegrityException) {
                Result.failure(workDataOf(Downloads.KEY_ERROR to error.message))
            } catch (error: IOException) {
                if (runAttemptCount < 8) Result.retry()
                else Result.failure(workDataOf(Downloads.KEY_ERROR to (error.message ?: applicationContext.getString(R.string.network_error))))
            }
        }
    }

    private suspend fun download(artifact: Artifact) {
        if (artifact.parts.isNotEmpty()) return downloadMerged(artifact)
        val context = applicationContext
        ModelFiles.dir(context, artifact).mkdirs()
        val part = ModelFiles.part(context, artifact)
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        var done = part.takeIf { it.isFile }?.length() ?: 0L
        if (done > artifact.sizeBytes) {
            part.delete()
            done = 0
        }
        if (!rehash(artifact, part, done, digest, buffer)) return
        requireStorage(artifact, done)
        if (done < artifact.sizeBytes) {
            val connection = open(artifact.url, done)
            try {
                val code = connection.responseCode
                if (done > 0 && code == HttpURLConnection.HTTP_OK) {
                    // The server ignored the range: restart from the beginning.
                    part.delete()
                    digest.reset()
                    done = 0
                } else if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL) {
                    throw IOException(applicationContext.getString(R.string.server_returned_http_1_d, code))
                }
                var lastBytes = done
                var lastTime = System.nanoTime()
                var speed = 0L
                connection.inputStream.use { input ->
                    FileOutputStream(part, done > 0).use { output ->
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            digest.update(buffer, 0, count)
                            done += count
                            if (done > artifact.sizeBytes) throw IntegrityException(applicationContext.getString(R.string.download_exceeds_the_expected_size))
                            val now = System.nanoTime()
                            if (now - lastTime >= 700_000_000L) {
                                speed = (done - lastBytes) * 1_000_000_000L / (now - lastTime)
                                lastBytes = done
                                lastTime = now
                                report(artifact, Downloads.PHASE_DOWNLOADING, done, artifact.sizeBytes, speed)
                            }
                            if (isStopped) return
                        }
                    }
                }
            } finally {
                connection.disconnect()
            }
        }
        if (done != artifact.sizeBytes) throw IOException(applicationContext.getString(R.string.connection_closed_early))
        finish(artifact, part, digest)
    }

    /** The hash state is not persisted, so a resumed transfer first re-reads what is already on disk. */
    private suspend fun rehash(artifact: Artifact, part: File, done: Long, digest: MessageDigest, buffer: ByteArray): Boolean {
        if (done == 0L) return true
        var hashed = 0L
        part.inputStream().use { input ->
            while (hashed < done) {
                val count = input.read(buffer, 0, minOf(buffer.size.toLong(), done - hashed).toInt())
                if (count < 0) break
                digest.update(buffer, 0, count)
                hashed += count
                report(artifact, Downloads.PHASE_VERIFYING, hashed, done, 0)
                if (isStopped) return false
            }
        }
        if (hashed != done) throw IOException(applicationContext.getString(R.string.the_partial_download_changed_while_it_was_checked))
        return true
    }

    private fun requireStorage(artifact: Artifact, done: Long) {
        val needed = artifact.sizeBytes - done + (1L shl 30)
        if (ModelFiles.freeBytes(applicationContext) < needed) {
            throw IntegrityException(applicationContext.getString(R.string.not_enough_free_storage_1_1f_gb_needed, needed / 1e9))
        }
    }

    private fun finish(artifact: Artifact, part: File, digest: MessageDigest) {
        val context = applicationContext
        val hash = digest.digest().joinToString("") { "%02x".format(it) }
        if (hash != artifact.sha256) {
            part.delete()
            throw IntegrityException(applicationContext.getString(R.string.checksum_mismatch_the_download_was_discarded))
        }
        if (!part.renameTo(ModelFiles.final(context, artifact))) throw IOException(applicationContext.getString(R.string.cannot_finalize_the_downloaded_file))
        ModelFiles.markVerified(context, artifact)
    }

    /**
     * Downloads gguf-split shards as their merge (see [GgufMerge]): the shards' headers come first (ranged
     * requests), the merged header is checked against its pinned hash and written, then each shard's data
     * section is appended as it streams. A resumed transfer re-hashes the partial file and continues inside the
     * shard where it stopped. The finished file must match the pinned SHA-256 of the whole merge.
     */
    private suspend fun downloadMerged(artifact: Artifact) {
        val context = applicationContext
        ModelFiles.dir(context, artifact).mkdirs()
        val part = ModelFiles.part(context, artifact)
        val shards = artifact.parts.map { shardHeader(artifact, it) ?: return }
        val header = GgufMerge.mergedHeader(shards)
        val headerHash = MessageDigest.getInstance("SHA-256").digest(header).joinToString("") { "%02x".format(it) }
        if (header.size.toLong() != artifact.headerBytes || headerHash != artifact.headerSha256 ||
            header.size + shards.sumOf { it.dataBytes } != artifact.sizeBytes) {
            throw IntegrityException(applicationContext.getString(R.string.the_repository_s_shards_do_not_match_the_pinned_model))
        }
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        var done = part.takeIf { it.isFile }?.length() ?: 0L
        // Only a partial file that starts with the whole merged header is resumed.
        if (done > artifact.sizeBytes || (done > 0 && !startsWith(part, header))) {
            part.delete()
            done = 0
        }
        if (!rehash(artifact, part, done, digest, buffer)) return
        requireStorage(artifact, done)
        FileOutputStream(part, done > 0).use { output ->
            if (done == 0L) {
                output.write(header)
                digest.update(header)
                done = header.size.toLong()
            }
            var start = header.size.toLong() // merged offset of this shard's data section
            for ((index, shard) in shards.withIndex()) {
                val end = start + shard.dataBytes
                if (done < end) {
                    val from = shard.dataOffset + (done - start)
                    done = transfer(artifact, artifact.url(artifact.parts[index].file), from, end, done, output, digest, buffer)
                        ?: return
                }
                start = end
            }
        }
        if (done != artifact.sizeBytes) throw IOException(applicationContext.getString(R.string.connection_closed_early))
        finish(artifact, part, digest)
    }

    private fun startsWith(file: File, prefix: ByteArray): Boolean {
        if (file.length() < prefix.size) return false
        val head = ByteArray(prefix.size)
        file.inputStream().use { input ->
            var read = 0
            while (read < head.size) {
                val count = input.read(head, read, head.size - read)
                if (count < 0) return false
                read += count
            }
        }
        return head.contentEquals(prefix)
    }

    /** Reads a shard's header with ranged requests, growing the range until the header is complete. */
    private fun shardHeader(artifact: Artifact, shard: ArtifactPart): GgufMerge.Shard? {
        var probe = minOf(shard.sizeBytes, 1L shl 20)
        while (true) {
            if (isStopped) return null
            val connection = open(artifact.url(shard.file), 0, probe - 1)
            val bytes = try {
                if (connection.responseCode != HttpURLConnection.HTTP_PARTIAL ||
                    connection.getHeaderField("Content-Range") != "bytes 0-${probe - 1}/${shard.sizeBytes}") {
                    throw IntegrityException(applicationContext.getString(R.string.the_server_did_not_return_the_requested_range_of_a_shard))
                }
                connection.inputStream.use { it.readBytes() }
            } finally {
                connection.disconnect()
            }
            if (bytes.size.toLong() != probe) throw IOException(applicationContext.getString(R.string.connection_closed_early))
            try {
                return GgufMerge.parse(bytes, shard.sizeBytes)
            } catch (_: GgufMerge.Truncated) {
                if (probe == shard.sizeBytes) throw IntegrityException(applicationContext.getString(R.string.a_shard_header_is_incomplete))
                probe = minOf(shard.sizeBytes, probe * 4)
            }
        }
    }

    /**
     * Appends the bytes of [url] from [from] until the merged file reaches [end]; returns the merged size, or null
     * when the work was stopped.
     */
    private suspend fun transfer(artifact: Artifact, url: String, from: Long, end: Long, start: Long,
                                 output: FileOutputStream, digest: MessageDigest, buffer: ByteArray): Long? {
        var done = start
        val connection = open(url, from)
        try {
            val code = connection.responseCode
            if (code != HttpURLConnection.HTTP_PARTIAL) throw IOException(applicationContext.getString(R.string.server_returned_http_1_d_for_a_shard_range, code))
            var lastBytes = done
            var lastTime = System.nanoTime()
            connection.inputStream.use { input ->
                while (done < end) {
                    val count = input.read(buffer, 0, minOf(buffer.size.toLong(), end - done).toInt())
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    digest.update(buffer, 0, count)
                    done += count
                    val now = System.nanoTime()
                    if (now - lastTime >= 700_000_000L) {
                        val speed = (done - lastBytes) * 1_000_000_000L / (now - lastTime)
                        lastBytes = done
                        lastTime = now
                        report(artifact, Downloads.PHASE_DOWNLOADING, done, artifact.sizeBytes, speed)
                    }
                    if (isStopped) return null
                }
            }
        } finally {
            connection.disconnect()
        }
        if (done != end) throw IOException(applicationContext.getString(R.string.connection_closed_early))
        return done
    }

    /** Follows redirects manually so the Range header reaches the final host; [to] < 0 reads to the end. */
    private fun open(url: String, from: Long, to: Long = -1): HttpURLConnection {
        var location = url
        repeat(6) {
            val connection = (URL(location).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "Equity/${BuildConfigInfo.VERSION}")
                if (from > 0 || to >= 0) setRequestProperty("Range", "bytes=$from-${if (to >= 0) to.toString() else ""}")
            }
            val code = connection.responseCode
            if (code in 300..399) {
                val next = connection.getHeaderField("Location") ?: throw IOException(applicationContext.getString(R.string.redirect_without_location))
                connection.disconnect()
                location = URL(URL(location), next).toString()
                if (!location.startsWith("https://")) throw IOException(applicationContext.getString(R.string.refusing_a_non_https_redirect))
            } else {
                return connection
            }
        }
        throw IOException(applicationContext.getString(R.string.too_many_redirects))
    }

    private suspend fun report(artifact: Artifact, phase: String, done: Long, total: Long, speed: Long) {
        setProgress(workDataOf(Downloads.KEY_PHASE to phase, Downloads.KEY_DONE to done,
            Downloads.KEY_TOTAL to total, Downloads.KEY_SPEED to speed))
        val label = if (phase == Downloads.PHASE_VERIFYING) applicationContext.getString(R.string.checking_the_partial_download)
            else applicationContext.getString(R.string.value_1_s_of_2_s, formatBytes(done), formatBytes(total))
        runCatching { setForeground(foregroundInfo(artifact, done, total, label)) }
    }

    private fun foregroundInfo(artifact: Artifact, done: Long, total: Long, label: String): ForegroundInfo {
        val manager = applicationContext.getSystemService(NotificationManager::class.java)
        run {
            manager.createNotificationChannel(NotificationChannel(CHANNEL, applicationContext.getString(R.string.model_downloads), NotificationManager.IMPORTANCE_LOW))
        }
        val percent = if (total > 0) (done * 100 / total).toInt() else 0
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(applicationContext.getString(R.string.downloading_1_s, artifact.name))
            .setContentText(label)
            .setProgress(100, percent, total <= 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .addAction(0, applicationContext.getString(R.string.pause), WorkManager.getInstance(applicationContext).createCancelPendingIntent(id))
            .build()
        val notificationId = NOTIFICATION_BASE + Catalog.artifacts.indexOf(artifact)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ForegroundInfo(notificationId, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            ForegroundInfo(notificationId, notification)
        }
    }

    private companion object {
        const val CHANNEL = "downloads"
        const val NOTIFICATION_BASE = 4100
    }
}

object BuildConfigInfo {
    const val VERSION = "0.2.0"
}

fun formatBytes(bytes: Long): String = when {
    bytes >= 1_000_000_000L -> "%.1f GB".format(bytes / 1e9)
    bytes >= 1_000_000L -> "%.0f MB".format(bytes / 1e6)
    else -> "%.0f kB".format(bytes / 1e3)
}
