package org.equity.app

import android.app.Instrumentation
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.concurrent.thread

class NativeSmokeTest : Instrumentation() {
    private var arguments = Bundle()
    override fun onCreate(arguments: Bundle?) {
        this.arguments = arguments ?: Bundle()
        super.onCreate(arguments)
        start()
    }
    override fun onStart() {
        try {
            val result = if (arguments.getString("case") == "attachment-context") testAttachmentContext()
                else if (arguments.getString("case") == "unload-cancel") testUnloadCancellation()
                else {
                    testRealInferenceReloadAndCancellation()
                    "inference, reload, cancellation, recovery"
                }
            finish(Activity.RESULT_OK, Bundle().apply { putString("stream", "PASS: $result\n") })
        } catch (error: Throwable) {
            finish(Activity.RESULT_CANCELED, Bundle().apply { putString("stream", "FAIL: ${error.stackTraceToString()}\n") })
        }
    }

    private fun testUnloadCancellation(): String {
        val app = targetContext.applicationContext
        val activity = startActivitySync(Intent(app, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        val saved = SettingsStore.state.value
        val previousChat = ChatStore.state.value.currentId
        var testChat: String? = null
        try {
            SettingsStore.update(app) { it.copy(context = 4096, maxTokens = 24, gpu = false) }
            runOnMainSync { EngineController.load(app, requireNotNull(Catalog.byId("qwen3-0.6b-q8"))) }
            awaitEngine { it.phase == Phase.Ready }
            ChatStore.newChat()
            runOnMainSync { check(EngineController.send(app, null, "hello ".repeat(3000), emptyList())) }
            testChat = requireNotNull(EngineController.state.value.generation).chatId
            Thread.sleep(100)
            val start = SystemClock.elapsedRealtime()
            runOnMainSync { EngineController.unload() }
            awaitEngine(timeoutMs = 10_000) { it.phase == Phase.Empty }
            val elapsed = SystemClock.elapsedRealtime() - start
            return "unloading cancels an in-flight prompt ($elapsed ms)"
        } finally {
            EngineController.unload()
            awaitEngine { it.phase == Phase.Empty }
            restoreSettings(app, saved)
            testChat?.let { ChatStore.delete(it) }
            if (previousChat != null) ChatStore.open(previousChat) else ChatStore.newChat()
            runOnMainSync { activity.finish() }
        }
    }

    private fun restoreSettings(context: Context, settings: Settings) {
        SettingsStore.update(context) { settings }
        // Instrumentation can end the process immediately; commit waits for outstanding apply writes.
        check(context.getSharedPreferences("settings", Context.MODE_PRIVATE).edit().commit())
    }

    private fun awaitEngine(timeoutMs: Long = 600_000, condition: (EngineState) -> Boolean): EngineState {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            val state = EngineController.state.value
            if (condition(state)) return state
            check(state.error == null || state.phase == Phase.Unloading) { state.error ?: "Engine failed" }
            Thread.sleep(100)
        }
        error("Timed out: ${EngineController.state.value.status}")
    }

    private fun testAttachmentContext(): String {
        val app = targetContext.applicationContext
        val activity = startActivitySync(Intent(app, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        val saved = SettingsStore.state.value
        val previousChat = ChatStore.state.value.currentId
        var testChat: String? = null
        val fixture = File(app.cacheDir, "context-regression.json")
        try {
            val original = arguments.getString("attachment_file")?.let { File(app.filesDir, it) }
            if (original != null) {
                check(original.canonicalPath.startsWith(app.filesDir.canonicalPath + File.separator))
                original.copyTo(fixture, overwrite = true)
            } else {
                fixture.writeText(JSONArray().apply {
                    repeat(350) { put(JSONObject().put("sample", it).put("description", "Attachment context regression")) }
                }.toString(2))
            }
            SettingsStore.update(app) { it.copy(context = 4096, maxTokens = 2048, temperature = 0f,
                gpu = arguments.getString("gpu") == "true", expertCacheMiB = 0) }
            val model = requireNotNull(Catalog.byId(arguments.getString("model") ?: "qwen3-0.6b-q8"))
            runOnMainSync { EngineController.load(app, model) }
            awaitEngine { it.phase == Phase.Ready }
            val attachments = Attachments.import(app, Uri.fromFile(fixture), vision = false)
            check(attachments.single().kind == AttachmentKind.Document)
            check(File(ChatStore.staging, attachments.single().file).readText() == fixture.readText().trim())
            ChatStore.newChat()
            runOnMainSync { check(EngineController.send(app, null, "What does this file contain?", attachments)) }
            val chatId = requireNotNull(EngineController.state.value.generation).chatId
            testChat = chatId
            awaitEngine {
                if (it.phase == Phase.Generating && (it.generation?.tokens ?: 0) >= 24) EngineController.stop()
                it.phase == Phase.Ready && it.generation == null
            }
            val result = JSONObject(requireNotNull(EngineController.state.value.lastResult))
            val promptTokens = result.getInt("prompt_tokens")
            check(promptTokens + 2048 > 4096) { "The fixture did not exercise context overflow" }
            val answer = ChatStore.state.value.loaded.getValue(chatId).messages.last()
            check(!answer.failed && answer.text.isNotBlank())
            check(result.getInt("generated_tokens") > 0)
            val expanded = requireNotNull(EngineController.state.value.configuration)
            if (model.id == "qwen3-0.6b-q8") {
                runOnMainSync {
                    check(EngineController.send(app, chatId, " hello".repeat(50_000), emptyList()))
                }
                awaitEngine { it.phase == Phase.Ready && it.generation == null }
                val rejected = ChatStore.state.value.loaded.getValue(chatId).messages.last()
                check(rejected.failed && rejected.text.contains("exceeding the model's"))
                runOnMainSync { check(EngineController.send(app, chatId, "Say hello.", emptyList())) }
                awaitEngine {
                    if (it.phase == Phase.Generating && (it.generation?.tokens ?: 0) >= 24) EngineController.stop()
                    it.phase == Phase.Ready && it.generation == null
                }
                check(!ChatStore.state.value.loaded.getValue(chatId).messages.last().failed)
            }
            return "attachment import, context growth, answer: $promptTokens input tokens; $expanded"
        } finally {
            EngineController.unload()
            awaitEngine { it.phase == Phase.Empty }
            restoreSettings(app, saved)
            testChat?.let { ChatStore.delete(it) }
            if (previousChat != null) ChatStore.open(previousChat) else ChatStore.newChat()
            fixture.delete()
            runOnMainSync { activity.finish() }
        }
    }
    fun testRealInferenceReloadAndCancellation() {
        val context = targetContext
        val installed = File(context.filesDir, "models/qwen3-0.6b-q8/Qwen3-0.6B-Q8_0.gguf")
        val model = installed.takeIf { it.isFile } ?: File(context.filesDir, "models/smoke.gguf")
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
            val oversized = JSONObject(request.toString()).put("messages", JSONArray().put(
                JSONObject().put("role", "user").put("content", " hello".repeat(600))))
            val overflow = runCatching { NativeBridge.generate(oversized.toString().toByteArray()) { } }.exceptionOrNull()
            check(overflow is ContextOverflowException && overflow.promptTokens + overflow.outputTokens > overflow.contextTokens)
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
