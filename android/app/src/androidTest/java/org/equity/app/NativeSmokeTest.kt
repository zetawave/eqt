package org.equity.app

import android.app.Instrumentation
import android.app.Activity
import android.os.Bundle
import android.os.ParcelFileDescriptor
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.concurrent.thread

class NativeSmokeTest : Instrumentation() {
    override fun onCreate(arguments: Bundle?) { super.onCreate(arguments); start() }
    override fun onStart() {
        try {
            testRealInferenceReloadAndCancellation()
            finish(Activity.RESULT_OK, Bundle().apply { putString("stream", "PASS: inference, reload, cancellation, recovery\n") })
        } catch (error: Throwable) {
            finish(Activity.RESULT_CANCELED, Bundle().apply { putString("stream", "FAIL: ${error.stackTraceToString()}\n") })
        }
    }
    fun testRealInferenceReloadAndCancellation() {
        val context = targetContext
        val model = File(context.filesDir, "models/smoke.gguf")
        model.parentFile!!.mkdirs()
        if (!model.isFile) {
            val partial = File(model.parentFile, "smoke.gguf.part")
            try {
                val source = uiAutomation.executeShellCommand("cat /data/local/tmp/eqt/model.gguf")
                ParcelFileDescriptor.AutoCloseInputStream(source).use { input ->
                    partial.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
                }
                check(partial.length() == 639446688L) { "Stage the official fixture with tools/benchmark.py first" }
                check(partial.renameTo(model))
            } finally { partial.delete() }
        }
        val digest = MessageDigest.getInstance("SHA-256")
        model.inputStream().use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        check(digest.digest().joinToString("") { "%02x".format(it) } ==
            "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031") { "Fixture SHA-256 mismatch" }
        val options = """{"context":512,"batch":32,"threads":2,"memory_budget_mib":2048}""".toByteArray()
        val request = JSONObject().put("messages", JSONArray().put(
            JSONObject().put("role", "user").put("content", "Say hello in Italian.")))
            .put("max_tokens", 24).put("temperature", 0).put("thinking", false)
        try {
            NativeBridge.prepare()
            ParcelFileDescriptor.open(model, ParcelFileDescriptor.MODE_READ_ONLY).use { descriptor ->
                NativeBridge.load("/proc/self/fd/${descriptor.fd}".toByteArray(), options) { }
            }
            val first = JSONObject(NativeBridge.generate(request.toString().toByteArray()) { }.decodeToString())
            check(first.getInt("generated_tokens") > 0)
            NativeBridge.unload()
            NativeBridge.prepare()
            NativeBridge.load(model.path.toByteArray(), options) { }
            val second = JSONObject(NativeBridge.generate(request.toString().toByteArray()) { }.decodeToString())
            check(first.getJSONArray("token_ids").toString() == second.getJSONArray("token_ids").toString())
            NativeBridge.prepare()
            request.put("max_tokens", 300).put("ignore_eos", true)
            val stop = thread { Thread.sleep(250); NativeBridge.cancel() }
            val cancelled = JSONObject(NativeBridge.generate(request.toString().toByteArray()) { }.decodeToString())
            stop.join()
            check(cancelled.getString("termination") == "cancelled")
            NativeBridge.prepare()
            request.put("max_tokens", 24).put("ignore_eos", false)
            val recovered = JSONObject(NativeBridge.generate(request.toString().toByteArray()) { }.decodeToString())
            check(first.getJSONArray("token_ids").toString() == recovered.getJSONArray("token_ids").toString())
            File(context.filesDir, "instrumentation-result.json").writeText(second.toString(2))
        } finally { NativeBridge.unload() }
    }
}
