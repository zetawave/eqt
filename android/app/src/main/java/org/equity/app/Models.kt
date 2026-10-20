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
    val tagline: String,
    val description: String,
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

    val models = listOf(
        CatalogModel(
            id = "qwen3-0.6b-q8", name = "Qwen3 0.6B", tagline = "Small · fully in memory · fast",
            description = "Official 8-bit build. Lightweight and quick; good for testing and short answers.",
            repository = "Qwen/Qwen3-0.6B-GGUF", revision = "23749fefcc72300e3a2ad315e1317431b06b590a",
            file = "Qwen3-0.6B-Q8_0.gguf", sizeBytes = 639_446_688L,
            sha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
            license = "Apache-2.0", streamed = false, residentMiB = 610, contextMiBPer1k = 112, badge = "0.6B",
        ),
        CatalogModel(
            id = "qwen3.6-35b-a3b-iq2m", name = "Qwen3.6 35B-A3B · 2.7-bit",
            tagline = "Large · twice the phone's RAM · fastest · experimental",
            description = "The same 35B mixture-of-experts model in Unsloth's dynamic 2.7-bit build (12 GB). On Equity's " +
                "27-prompt quality set it scored like the 4-bit build and ran about a third faster: around 5 tokens/s " +
                "on a cool phone, about 3.5 sustained once it warms up. Experts are read from storage on demand.",
            repository = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", revision = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
            file = "Qwen3.6-35B-A3B-UD-IQ2_M.gguf", sizeBytes = 11_882_969_376L,
            sha256 = "b989da14b63b29bd469cb1fab5bd16b422ab921787569b4b2bb4fd61e26129b8",
            license = "Apache-2.0", streamed = true, residentMiB = 1430, contextMiBPer1k = 40,
            projector = qwen36Projector, badge = "35B",
            stream = StreamProfile(layerExpertsMiB = 240, experts = 256, expertsUsed = 8),
        ),
        CatalogModel(
            id = "qwen3.6-35b-a3b-q4km", name = "Qwen3.6 35B-A3B · 4-bit",
            tagline = "Large · twice the phone's RAM · quality reference · experimental",
            description = "The same 35B mixture-of-experts model in Unsloth's dynamic 4-bit build (22.7 GB), Equity's " +
                "quality reference. Shared weights stay in memory and experts are read from storage on demand. " +
                "Expect roughly 3 tokens/s; the phone warms up during long answers.",
            repository = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", revision = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
            file = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", sizeBytes = 22_663_387_424L,
            sha256 = "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b",
            license = "Apache-2.0", streamed = true, residentMiB = 1935, contextMiBPer1k = 40,
            projector = qwen36Projector, badge = "35B",
            stream = StreamProfile(layerExpertsMiB = 466, experts = 256, expertsUsed = 8),
        ),
        // Unsloth splits the 78.9 GB build into three gguf-split shards (configs/flash-next-model.json); the app
        // writes their merge, byte-identical to llama-gguf-split --merge of the same shards.
        CatalogModel(
            id = "qwen3.8-flash-next-ud-q2kxl", name = "Qwen3.8 Flash Next · Q2_K_XL",
            tagline = "Huge · seven times the phone's RAM · experimental",
            description = "Qwen's 125B mixture-of-experts preview of the Qwen4 architecture (512 experts, 10 per token, " +
                "plus a 51B n-gram memory) in Unsloth's dynamic UD-Q2_K_XL build, experts at about 3 bits (78.9 GB). " +
                "Only 3.4 GB stays in memory; " +
                "experts and n-gram rows are read from storage as each token needs them, so answers start slowly " +
                "and run at a few tokens per second. Needs about 80 GB of free storage.",
            repository = "unsloth/Qwen3.8-Flash-Next-GGUF", revision = "766911a6b7369840a91dbcd95f9f997acaab6cd6",
            file = "Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf", sizeBytes = 78_869_128_640L,
            sha256 = "f26a5e679710fb4cd048f481ff082d9af9c79fbd640d1867ef58c425518659d7",
            // Shared weights 3,372 MiB, recurrent state 113 MiB; the expert banks and n-gram table are file-backed.
            license = "Qwen Community 1.0", streamed = true, residentMiB = 3500, contextMiBPer1k = 30,
            badge = "125B",
            // Compute buffers measured on the host: 518 MiB at 512 tokens and 4,096 context, 534 at 16,384; linear
            // in the batch. 512 experts: a prompt shorter than ~118 tokens uses under 90% of a layer's experts, so
            // reading the next layer whole wastes reads (52 tokens: 54 GB read instead of 31, measured on the host).
            // Lookahead reads in a pool beside the cache, two layers ahead (D029, Galaxy): 1.47 -> 1.68 tokens/s
            // at 1 GiB, 1.92 at 1.75 GiB, identical tokens. The pool holds about 60 MiB within the reserve.
            // Demand reads first and at most 3 speculative reads in flight (D030, Galaxy, 1.75 GiB): 2.12 -> 2.39.
            stream = StreamProfile(layerExpertsMiB = 916, experts = 512, expertsUsed = 10, computeMiBPer512 = 540,
                tuning = mapOf("prefill_prefetch_min_coverage" to 0.9, "speculative_pool" to true,
                    "lookahead_layers" to 2, "lookahead2_count" to 5, "read_priority" to true, "prefetch_inflight" to 3)),
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
            ?: return Result.failure(workDataOf(Downloads.KEY_ERROR to "Unknown file"))
        if (ModelFiles.installed(applicationContext, artifact)) return Result.success()
        // A foreground start can be refused when a retry runs in the background; the transfer still proceeds.
        runCatching { setForeground(foregroundInfo(artifact, 0, artifact.sizeBytes, "Starting")) }
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
                else Result.failure(workDataOf(Downloads.KEY_ERROR to (error.message ?: "Network error")))
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
                    throw IOException("Server returned HTTP $code")
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
                            if (done > artifact.sizeBytes) throw IntegrityException("Download exceeds the expected size")
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
        if (done != artifact.sizeBytes) throw IOException("Connection closed early")
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
        if (hashed != done) throw IOException("The partial download changed while it was checked")
        return true
    }

    private fun requireStorage(artifact: Artifact, done: Long) {
        val needed = artifact.sizeBytes - done + (1L shl 30)
        if (ModelFiles.freeBytes(applicationContext) < needed) {
            throw IntegrityException("Not enough free storage: %.1f GB needed".format(needed / 1e9))
        }
    }

    private fun finish(artifact: Artifact, part: File, digest: MessageDigest) {
        val context = applicationContext
        val hash = digest.digest().joinToString("") { "%02x".format(it) }
        if (hash != artifact.sha256) {
            part.delete()
            throw IntegrityException("Checksum mismatch; the download was discarded")
        }
        if (!part.renameTo(ModelFiles.final(context, artifact))) throw IOException("Cannot finalize the downloaded file")
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
            throw IntegrityException("The repository's shards do not match the pinned model")
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
        if (done != artifact.sizeBytes) throw IOException("Connection closed early")
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
                    throw IntegrityException("The server did not return the requested range of a shard")
                }
                connection.inputStream.use { it.readBytes() }
            } finally {
                connection.disconnect()
            }
            if (bytes.size.toLong() != probe) throw IOException("Connection closed early")
            try {
                return GgufMerge.parse(bytes, shard.sizeBytes)
            } catch (_: GgufMerge.Truncated) {
                if (probe == shard.sizeBytes) throw IntegrityException("A shard header is incomplete")
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
            if (code != HttpURLConnection.HTTP_PARTIAL) throw IOException("Server returned HTTP $code for a shard range")
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
        if (done != end) throw IOException("Connection closed early")
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
                val next = connection.getHeaderField("Location") ?: throw IOException("Redirect without location")
                connection.disconnect()
                location = URL(URL(location), next).toString()
                if (!location.startsWith("https://")) throw IOException("Refusing a non-HTTPS redirect")
            } else {
                return connection
            }
        }
        throw IOException("Too many redirects")
    }

    private suspend fun report(artifact: Artifact, phase: String, done: Long, total: Long, speed: Long) {
        setProgress(workDataOf(Downloads.KEY_PHASE to phase, Downloads.KEY_DONE to done,
            Downloads.KEY_TOTAL to total, Downloads.KEY_SPEED to speed))
        val label = if (phase == Downloads.PHASE_VERIFYING) "Checking the partial download"
            else "%s of %s".format(formatBytes(done), formatBytes(total))
        runCatching { setForeground(foregroundInfo(artifact, done, total, label)) }
    }

    private fun foregroundInfo(artifact: Artifact, done: Long, total: Long, label: String): ForegroundInfo {
        val manager = applicationContext.getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL) == null) {
            manager.createNotificationChannel(NotificationChannel(CHANNEL, "Model downloads", NotificationManager.IMPORTANCE_LOW))
        }
        val percent = if (total > 0) (done * 100 / total).toInt() else 0
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("Downloading ${artifact.name}")
            .setContentText(label)
            .setProgress(100, percent, total <= 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .addAction(0, "Pause", WorkManager.getInstance(applicationContext).createCancelPendingIntent(id))
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
