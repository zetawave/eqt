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

/** A pinned, hash-verified model artifact the app can download and run. */
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
    /** Approximate resident weight footprint in MiB, used for RAM admission. */
    val residentMiB: Int,
    /** KV cache plus recurrent state per 1024 context tokens, in MiB (F16 cache). */
    val contextMiBPer1k: Int,
) {
    val url get() = "https://huggingface.co/$repository/resolve/$revision/$file"
}

object Catalog {
    val models = listOf(
        CatalogModel(
            id = "qwen3-0.6b-q8", name = "Qwen3 0.6B", tagline = "Small · fully in memory · fast",
            description = "Official 8-bit build. Lightweight and quick; good for testing and short answers.",
            repository = "Qwen/Qwen3-0.6B-GGUF", revision = "23749fefcc72300e3a2ad315e1317431b06b590a",
            file = "Qwen3-0.6B-Q8_0.gguf", sizeBytes = 639_446_688L,
            sha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
            license = "Apache-2.0", streamed = false, residentMiB = 610, contextMiBPer1k = 112,
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
            license = "Apache-2.0", streamed = true, residentMiB = 1760, contextMiBPer1k = 40,
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
            license = "Apache-2.0", streamed = true, residentMiB = 2448, contextMiBPer1k = 40,
        ),
    )

    fun byId(id: String?) = models.firstOrNull { it.id == id }
}

object ModelFiles {
    fun dir(context: Context, model: CatalogModel) = File(context.filesDir, "models/${model.id}")
    fun final(context: Context, model: CatalogModel) = File(dir(context, model), model.file)
    fun part(context: Context, model: CatalogModel) = File(dir(context, model), model.file + ".part")
    private fun marker(context: Context, model: CatalogModel) = File(dir(context, model), model.file + ".sha256")

    /** True only after the full download matched the pinned SHA-256. */
    fun installed(context: Context, model: CatalogModel): Boolean {
        val file = final(context, model)
        val marker = marker(context, model)
        return file.isFile && file.length() == model.sizeBytes && marker.isFile && marker.readText().trim() == model.sha256
    }

    fun partialBytes(context: Context, model: CatalogModel) = part(context, model).takeIf { it.isFile }?.length() ?: 0L

    fun markVerified(context: Context, model: CatalogModel) = marker(context, model).writeText(model.sha256)

    fun delete(context: Context, model: CatalogModel) = dir(context, model).deleteRecursively()

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

    private fun name(model: CatalogModel) = "download-${model.id}"

    fun start(context: Context, model: CatalogModel) {
        // The large artifact waits for an unmetered network instead of consuming mobile data.
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(if (model.streamed) NetworkType.UNMETERED else NetworkType.CONNECTED)
            .setRequiresStorageNotLow(true)
            .build()
        val request = OneTimeWorkRequestBuilder<DownloadWorker>()
            .setInputData(workDataOf(KEY_ID to model.id))
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .addTag("model-download")
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(name(model), ExistingWorkPolicy.REPLACE, request)
    }

    /** Stops the transfer and keeps the partial file for a later resume. */
    fun pause(context: Context, model: CatalogModel) {
        WorkManager.getInstance(context).cancelUniqueWork(name(model))
    }

    /** Cancels any transfer, waits for the worker to release the file, then removes all local data. */
    fun delete(context: Context, model: CatalogModel, done: () -> Unit) {
        val app = context.applicationContext
        val manager = WorkManager.getInstance(app)
        manager.cancelUniqueWork(name(model))
        Thread {
            runCatching {
                while (manager.getWorkInfosForUniqueWork(name(model)).get().any { !it.state.isFinished }) Thread.sleep(100)
            }
            ModelFiles.delete(app, model)
            done()
        }.start()
    }

    fun observe(context: Context, model: CatalogModel): Flow<WorkInfo?> =
        WorkManager.getInstance(context).getWorkInfosForUniqueWorkFlow(name(model)).map { it.firstOrNull() }
}

/** Resumable, hash-verified download of one catalog model, run as a data-sync foreground job. */
class DownloadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    private class IntegrityException(message: String) : Exception(message)

    override suspend fun doWork(): Result {
        val model = Catalog.byId(inputData.getString(Downloads.KEY_ID))
            ?: return Result.failure(workDataOf(Downloads.KEY_ERROR to "Unknown model"))
        if (ModelFiles.installed(applicationContext, model)) return Result.success()
        // A foreground start can be refused when a retry runs in the background; the transfer still proceeds.
        runCatching { setForeground(foregroundInfo(model, 0, model.sizeBytes, "Starting")) }
        return withContext(Dispatchers.IO) {
            try {
                download(model)
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

    private suspend fun download(model: CatalogModel) {
        val context = applicationContext
        ModelFiles.dir(context, model).mkdirs()
        val part = ModelFiles.part(context, model)
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        var done = part.takeIf { it.isFile }?.length() ?: 0L
        if (done > model.sizeBytes) {
            part.delete()
            done = 0
        }
        // The hash state is not persisted, so a resumed transfer first re-reads what is already on disk.
        if (done > 0) {
            var hashed = 0L
            part.inputStream().use { input ->
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                    hashed += count
                    report(model, Downloads.PHASE_VERIFYING, hashed, done, 0)
                    if (isStopped) return
                }
            }
        }
        val needed = model.sizeBytes - done + (1L shl 30)
        if (ModelFiles.freeBytes(context) < needed) {
            throw IntegrityException("Not enough free storage: %.1f GB needed".format(needed / 1e9))
        }
        if (done < model.sizeBytes) {
            val connection = open(model.url, done)
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
                            if (done > model.sizeBytes) throw IntegrityException("Download exceeds the expected size")
                            val now = System.nanoTime()
                            if (now - lastTime >= 700_000_000L) {
                                speed = (done - lastBytes) * 1_000_000_000L / (now - lastTime)
                                lastBytes = done
                                lastTime = now
                                report(model, Downloads.PHASE_DOWNLOADING, done, model.sizeBytes, speed)
                            }
                            if (isStopped) return
                        }
                    }
                }
            } finally {
                connection.disconnect()
            }
        }
        if (done != model.sizeBytes) throw IOException("Connection closed early")
        val hash = digest.digest().joinToString("") { "%02x".format(it) }
        if (hash != model.sha256) {
            part.delete()
            throw IntegrityException("Checksum mismatch; the download was discarded")
        }
        if (!part.renameTo(ModelFiles.final(context, model))) throw IOException("Cannot finalize the model file")
        ModelFiles.markVerified(context, model)
    }

    /** Follows redirects manually so the Range header reaches the final host. */
    private fun open(url: String, from: Long): HttpURLConnection {
        var location = url
        repeat(6) {
            val connection = (URL(location).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "Equity/${BuildConfigInfo.VERSION}")
                if (from > 0) setRequestProperty("Range", "bytes=$from-")
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

    private suspend fun report(model: CatalogModel, phase: String, done: Long, total: Long, speed: Long) {
        setProgress(workDataOf(Downloads.KEY_PHASE to phase, Downloads.KEY_DONE to done,
            Downloads.KEY_TOTAL to total, Downloads.KEY_SPEED to speed))
        val label = if (phase == Downloads.PHASE_VERIFYING) "Checking the partial download"
            else "%s of %s".format(formatBytes(done), formatBytes(total))
        runCatching { setForeground(foregroundInfo(model, done, total, label)) }
    }

    private fun foregroundInfo(model: CatalogModel, done: Long, total: Long, label: String): ForegroundInfo {
        val manager = applicationContext.getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL) == null) {
            manager.createNotificationChannel(NotificationChannel(CHANNEL, "Model downloads", NotificationManager.IMPORTANCE_LOW))
        }
        val percent = if (total > 0) (done * 100 / total).toInt() else 0
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("Downloading ${model.name}")
            .setContentText(label)
            .setProgress(100, percent, total <= 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .addAction(0, "Pause", WorkManager.getInstance(applicationContext).createCancelPendingIntent(id))
            .build()
        val notificationId = NOTIFICATION_BASE + Catalog.models.indexOf(model)
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
