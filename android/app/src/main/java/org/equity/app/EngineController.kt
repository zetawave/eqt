package org.equity.app

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.PowerManager
import android.system.Os
import android.system.OsConstants
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.security.MessageDigest
import java.util.concurrent.Executors

object EngineController {
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private var descriptor: ParcelFileDescriptor? = null
    private var history = JSONArray()
    var observer: (() -> Unit)? = null
    var busy = false
        private set
    var loaded = false
        private set
    var status = "Select a local GGUF model. CPU backend. No network access."
        private set
    var configuration = "No model loaded"
        private set
    var transcript = ""
        private set
    var lastResult: String? = null
        private set
    private var modelHash = ""
    private var pendingTasks = 0
    private fun publish(action: () -> Unit) = main.post { action(); observer?.invoke() }

    private fun task(action: () -> Unit) {
        if (busy) return
        pendingTasks++
        busy = true
        NativeBridge.prepare()
        observer?.invoke()
        worker.execute {
            try { action() }
            catch (error: Exception) { publish { status = error.message ?: error.javaClass.simpleName } }
            finally { publish { pendingTasks--; busy = pendingTasks > 0 } }
        }
    }

    fun load(context: Context, uri: Uri?, options: JSONObject) {
        val app = context.applicationContext
        task {
            NativeBridge.unload()
            descriptor?.close()
            descriptor = null
            publish { loaded = false; configuration = "No model loaded"; status = "Checking model and memory…" }
            val privateModel = File(app.filesDir, "models/smoke.gguf")
            val pfd = if (uri != null) app.contentResolver.openFileDescriptor(uri, "r")
                else ParcelFileDescriptor.open(if (privateModel.canRead()) privateModel else
                    File(app.getExternalFilesDir(null), "models/smoke.gguf"), ParcelFileDescriptor.MODE_READ_ONLY)
            requireNotNull(pfd) { "Cannot open model" }
            descriptor = pfd
            try {
                val stat = Os.fstat(pfd.fileDescriptor)
                require(OsConstants.S_ISREG(stat.st_mode) && stat.st_size > 0) {
                    "Choose a local, seekable file. Cloud providers and pipes are unsupported."
                }
                val magic = ByteArray(4)
                require(Os.pread(pfd.fileDescriptor, magic, 0, 4, 0) == 4 && magic.decodeToString() == "GGUF") {
                    "The selected file is not GGUF"
                }
                val memory = ActivityManager.MemoryInfo()
                app.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
                val budget = options.getLong("memory_budget_mib") * 1024 * 1024
                require(!memory.lowMemory && budget + 512L * 1024 * 1024 <= memory.availMem) {
                    "Not enough available RAM for this budget plus 512 MiB system headroom; lower the budget or close apps manually."
                }
                require(stat.st_size <= budget - 512L * 1024 * 1024) { "Model exceeds resident budget. Expert streaming is not available yet." }
                publish { status = "Hashing ${stat.st_size / (1024 * 1024)} MiB without copying…" }
                val digest = MessageDigest.getInstance("SHA-256")
                val buffer = ByteArray(1024 * 1024)
                var offset = 0L
                while (offset < stat.st_size) {
                    if (cancelHash) throw IllegalStateException("Loading cancelled")
                    val count = Os.pread(pfd.fileDescriptor, buffer, 0, minOf(buffer.size.toLong(), stat.st_size - offset).toInt(), offset)
                    require(count > 0) { "Incomplete model read" }
                    digest.update(buffer, 0, count)
                    offset += count
                }
                modelHash = digest.digest().joinToString("") { "%02x".format(it) }
                val result = NativeBridge.load("/proc/self/fd/${pfd.fd}".toByteArray(), options.toString().toByteArray()) {
                    val percent = it.decodeToString()
                    publish { status = "Loading $percent%…" }
                }
                val info = JSONObject(result.decodeToString())
                history = JSONArray()
                publish {
                    loaded = true
                    transcript = ""
                    lastResult = null
                    configuration = "CPU · ${info.getInt("threads")} threads · context ${info.getInt("context")} · admission ${info.getInt("memory_budget_mib")} MiB"
                    status = "Ready · no expert cache"
                }
            } catch (error: Exception) {
                NativeBridge.unload()
                descriptor?.close()
                descriptor = null
                throw error
            }
        }
    }

    @Volatile private var cancelHash = false
    fun prepareLoad() { cancelHash = false }
    fun stop() { cancelHash = true; NativeBridge.cancel() }
    fun memoryPressure() {
        if (!busy && !loaded) return
        stop()
        pendingTasks++
        busy = true
        worker.execute {
            try {
                NativeBridge.unload()
                descriptor?.close()
                descriptor = null
            } finally {
                publish {
                    loaded = false
                    configuration = "No model loaded"
                    status = "Released model after memory pressure"
                    pendingTasks--
                    busy = pendingTasks > 0
                }
            }
        }
    }
    fun unload() = task {
        NativeBridge.unload()
        descriptor?.close()
        descriptor = null
        history = JSONArray()
        publish { loaded = false; configuration = "No model loaded"; status = "Model unloaded"; transcript = ""; lastResult = null }
    }
    fun newChat() {
        if (busy) return
        history = JSONArray()
        transcript = ""
        lastResult = null
        observer?.invoke()
    }

    fun generate(context: Context, prompt: String, system: String, maxTokens: Int, temperature: Double, benchmark: Boolean) {
        if (!loaded || busy) return
        val app = context.applicationContext
        val previousText = if (benchmark) "" else transcript
        task {
            val messages = JSONArray().put(JSONObject().put("role", "system").put("content", system))
            if (!benchmark) for (i in 0 until history.length()) messages.put(history.getJSONObject(i))
            messages.put(JSONObject().put("role", "user").put("content", prompt))
            val request = JSONObject().put("messages", messages).put("max_tokens", maxTokens)
                .put("temperature", temperature).put("seed", 42).put("thinking", false).put("ignore_eos", benchmark)
            val power = app.getSystemService(PowerManager::class.java)
            val thermalBefore = if (android.os.Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null
            val output = ByteArrayOutputStream()
            var lastUpdate = 0L
            publish { status = "Prefill…"; lastResult = null; transcript = "$previousText\nYou: $prompt\nEquity: " }
            val result = NativeBridge.generate(request.toString().toByteArray()) { chunk ->
                output.write(chunk)
                val now = android.os.SystemClock.elapsedRealtime()
                if (now - lastUpdate >= 50) {
                    lastUpdate = now
                    val decoder = Charsets.UTF_8.newDecoder()
                    val chars = CharBuffer.allocate(output.size())
                    decoder.decode(ByteBuffer.wrap(output.toByteArray()), chars, false)
                    chars.flip()
                    val text = chars.toString()
                    publish { transcript = "$previousText\nYou: $prompt\nEquity: $text"; status = "Generating…" }
                }
            }
            val json = JSONObject(result.decodeToString())
            json.put("surface", "app").put("model_sha256", modelHash)
                .put("workload_sha256", MessageDigest.getInstance("SHA-256").digest(request.toString().toByteArray()).joinToString("") { "%02x".format(it) })
                .put("thermal_status_before", thermalBefore ?: JSONObject.NULL)
                .put("thermal_status_after", if (android.os.Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else JSONObject.NULL)
            val answer = json.getString("text")
            if (!benchmark && json.getString("termination") != "cancelled") {
                history.put(JSONObject().put("role", "user").put("content", prompt))
                history.put(JSONObject().put("role", "assistant").put("content", answer))
            }
            publish {
                transcript = "$previousText\nYou: $prompt\nEquity: $answer\n"
                lastResult = json.toString(2)
                val ttft = if (json.isNull("ttft_ms")) "n/a" else "%.0f".format(java.util.Locale.US, json.getDouble("ttft_ms"))
                status = "%.2f tok/s · TTFT %s ms · %s · thermal %s".format(java.util.Locale.US,
                    json.getDouble("decode_tokens_per_second"), ttft, json.getString("termination"), json.opt("thermal_status_after"))
            }
        }
    }
}
