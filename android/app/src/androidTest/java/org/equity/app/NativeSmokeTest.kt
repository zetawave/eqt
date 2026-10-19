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
                else if (arguments.getString("case") == "steering") testSteeringImportAndChat()
                else if (arguments.getString("case") == "steering-real") testSteeringImportAndChat(realPackage = true)
                else if (arguments.getString("case") == "chat-ui") ChatUiChecks(this).run()
                else {
                    testRealInferenceReloadAndCancellation()
                    "inference, reload, cancellation, recovery"
                }
            finish(Activity.RESULT_OK, Bundle().apply { putString("stream", "PASS: $result\n") })
        } catch (error: Throwable) {
            finish(Activity.RESULT_CANCELED, Bundle().apply { putString("stream", "FAIL: ${error.stackTraceToString()}\n") })
        }
    }

    private fun testSteeringImportAndChat(realPackage: Boolean = false): String {
        val app = targetContext.applicationContext
        val activity = startActivitySync(Intent(app, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        val saved = SettingsStore.state.value
        val previousChat = ChatStore.state.value.currentId
        val previousDraft = ChatStore.state.value.steering
        val model = requireNotNull(Catalog.byId(if (realPackage) {
            arguments.getString("model") ?: "qwen3.6-35b-a3b-iq2m"
        } else "qwen3-0.6b-q8"))
        val gpu = arguments.getString("gpu") == "true"
        val growContext = arguments.getString("grow_context") == "true"
        val initialContext = if (growContext) 128 else 2048
        val strength = if (realPackage) 1f else -0.5f
        val index = File(app.filesDir, "steering/${model.sha256}/current.json")
        val previousIndex = index.takeIf { it.isFile }?.readBytes()
        val fixture = File(app.filesDir, if (realPackage) "bench/steering-real-vector.gguf" else "bench/steering-test-vector.gguf")
        val metadata = File(fixture.path + ".json")
        check(fixture.isFile && metadata.isFile) { "Stage the steering vector and matching JSON in files/bench first" }
        val vector = File(app.cacheDir, "steering-import-test.gguf")
        val manifest = File(app.cacheDir, "steering-import-test.gguf.json")
        val testChats = mutableListOf<String>()
        var completed = false
        try {
            fixture.copyTo(vector, overwrite = true)
            val source = JSONObject(metadata.readText())
            if (realPackage) check(source.getString("model_sha256") == model.sha256) { "Real package belongs to another model" }
            else source.put("model_sha256", model.sha256)
            manifest.writeText(source.toString())
            SettingsStore.update(app) { it.copy(systemPrompt = "You are a helpful assistant.", maxTokens = 8,
                temperature = 0f, context = initialContext, gpu = gpu, expertCacheMiB = 0) }
            runOnMainSync { EngineController.load(app, model) }
            val ready = awaitEngine { it.phase == Phase.Ready }
            runOnMainSync { EngineController.importSteering(app, listOf(Uri.fromFile(manifest), Uri.fromFile(vector))) }
            val imported = awaitEngine { it.phase == Phase.Ready && it.steeringProfile != null }
            check(imported.steeringProfile?.modelHash == model.sha256)
            check(imported.steeringProfile?.vectorHash == source.getString("vector_sha256"))
            check(imported.status == ready.status && !imported.busy)
            val prompt = if (realPackage || growContext) "The garden contains trees, flowers and a wooden bench. ".repeat(24) +
                "Summarize this garden in one short sentence." else "Say hello in one short sentence."
            fun sendFresh(options: SteeringOptions, text: String = prompt): JSONObject {
                runOnMainSync {
                    ChatStore.newChat()
                    check(!ChatStore.state.value.steering.enabled)
                    ChatStore.setSteering(options)
                    check(EngineController.send(app, null, text, emptyList()))
                }
                testChats += requireNotNull(EngineController.state.value.generation).chatId
                return JSONObject(requireNotNull(awaitEngine { it.phase == Phase.Ready }.lastResult)).also {
                    check(it.getInt("generated_tokens") > 0)
                    check(!requireNotNull(ChatStore.state.value.current).messages.last().failed)
                }
            }
            val baseline = if (realPackage) sendFresh(SteeringOptions()) else null
            runOnMainSync {
                ChatStore.newChat()
                check(!ChatStore.state.value.steering.enabled)
                ChatStore.setSteering(SteeringOptions(enabled = true, strength = strength))
                check(EngineController.send(app, null, prompt, emptyList()))
            }
            testChats += requireNotNull(EngineController.state.value.generation).chatId
            val steered = awaitEngine { it.phase == Phase.Ready }
            val on = JSONObject(requireNotNull(steered.lastResult))
            check(on.getJSONObject("steering").getDouble("strength") == strength.toDouble())
            check(on.getJSONObject("steering").getString("mode") == if (strength > 0) "addition" else "ablation")
            check(on.isNull("speculative"))
            check(on.getInt("generated_tokens") > 0)
            if (growContext) {
                check(on.getInt("prompt_tokens") + 8 > initialContext)
                check(on.getJSONObject("configuration").getInt("context") > initialContext)
                check(steered.steeringProfile?.vectorHash == source.getString("vector_sha256"))
            }
            val chat = requireNotNull(ChatStore.state.value.current)
            check(chat.steering.enabled && chat.steering.strength == strength)
            check(!chat.messages.last().failed && chat.messages.last().steering != null)
            val disk = File(app.filesDir, "chats/${chat.id}/chat.json")
            val deadline = SystemClock.elapsedRealtime() + 5000
            while (SystemClock.elapsedRealtime() < deadline && runCatching {
                    val json = JSONObject(disk.readText())
                    json.optBoolean("steering_enabled") && json.optDouble("steering_strength") == strength.toDouble()
                }.getOrDefault(false).not()) Thread.sleep(50)
            check(JSONObject(disk.readText()).getBoolean("steering_enabled"))
            check(JSONObject(disk.readText()).getDouble("steering_strength") == strength.toDouble())
            runOnMainSync {
                ChatStore.newChat()
                check(!ChatStore.state.value.steering.enabled)
                ChatStore.open(chat.id)
                check(ChatStore.state.value.steering.enabled)
                ChatStore.setSteering(SteeringOptions(enabled = false, strength = strength))
            }
            var cancellation: JSONObject? = null
            if (realPackage) {
                SettingsStore.update(app) { it.copy(maxTokens = 64) }
                runOnMainSync {
                    ChatStore.newChat()
                    ChatStore.setSteering(SteeringOptions(enabled = true, strength = -0.5f))
                    check(EngineController.send(app, null, "Count from one to twenty, one number per line.", emptyList()))
                }
                testChats += requireNotNull(EngineController.state.value.generation).chatId
                val stopped = awaitEngine {
                    if (it.phase == Phase.Generating && (it.generation?.tokens ?: 0) >= 1) EngineController.stop()
                    it.phase == Phase.Ready
                }
                cancellation = JSONObject(requireNotNull(stopped.lastResult))
                check(cancellation.getString("termination") == "cancelled")
                check(cancellation.getJSONObject("steering").getDouble("strength") == -0.5)
                check(cancellation.getJSONObject("steering").getString("mode") == "ablation")
                SettingsStore.update(app) { it.copy(maxTokens = 8) }
            }
            val plain = sendFresh(SteeringOptions())
            check(plain.isNull("steering"))
            if (baseline != null) {
                check(plain.getJSONArray("token_ids").toString() == baseline.getJSONArray("token_ids").toString()) {
                    "Steering OFF did not recover baseline tokens"
                }
                val offload = on.optJSONObject("gpu_offload")
                if (gpu) check(offload != null && !offload.getBoolean("failed") && offload.getLong("dense_ops") > 0)
                File(app.filesDir, "bench/steering-integration-${model.id}-${if (gpu) "gpu" else "cpu"}.json").writeText(
                    JSONObject().put("passed", true).put("model_sha256", model.sha256)
                        .put("steering", on.getJSONObject("steering")).put("gpu", gpu)
                        .put("gpu_offload", offload ?: JSONObject.NULL).put("prompt_tokens", on.getInt("prompt_tokens"))
                        .put("generated_tokens", on.getInt("generated_tokens")).put("off_tokens_equal", true)
                        .put("cancelled_strength", requireNotNull(cancellation).getJSONObject("steering").getDouble("strength"))
                        .put("chat_persistence", true).put("evaluation", "functional only").toString(2))
            }
            if (growContext) File(app.filesDir, "bench/steering-context-integration.json").writeText(
                JSONObject().put("passed", true).put("initial_context", initialContext)
                    .put("expanded_context", on.getJSONObject("configuration").getInt("context"))
                    .put("prompt_tokens", on.getInt("prompt_tokens")).put("strength", strength.toDouble())
                    .put("steering_preserved", true).put("off_after_expansion", plain.isNull("steering"))
                    .put("evaluation", "synthetic direction; functional only").toString(2))
            completed = true
            return "vector/JSON import, model binding, per-chat strength persistence, steering ON and OFF"
        } finally {
            runOnMainSync { EngineController.unload() }
            awaitEngine { it.phase == Phase.Empty }
            restoreSettings(app, saved)
            testChats.forEach { ChatStore.delete(it) }
            if (!(completed && realPackage && arguments.getString("keep_vector") == "true")) {
                if (previousIndex == null) index.delete() else {
                    index.parentFile?.mkdirs()
                    index.writeBytes(previousIndex)
                }
            }
            vector.delete()
            manifest.delete()
            if (previousChat != null) ChatStore.open(previousChat) else {
                ChatStore.newChat()
                ChatStore.setSteering(previousDraft)
            }
            runOnMainSync { activity.finish() }
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
