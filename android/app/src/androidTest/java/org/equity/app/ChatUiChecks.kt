package org.equity.app

import android.app.Instrumentation
import android.app.Notification
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.equity.app.ui.AssistantMessage
import org.equity.app.ui.ChatScreen
import org.equity.app.ui.EquityTheme
import org.json.JSONObject
import java.io.File

internal class ChatUiChecks(private val instrumentation: Instrumentation) {
    private fun find(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
        fun visit(node: AccessibilityNodeInfo): AccessibilityNodeInfo? {
            if (predicate(node)) return node
            for (index in 0 until node.childCount) node.getChild(index)?.let { child ->
                visit(child)?.let { return it }
            }
            return null
        }
        return instrumentation.uiAutomation.rootInActiveWindow?.let(::visit)
    }

    private fun awaitNode(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo {
        val deadline = SystemClock.elapsedRealtime() + 10_000
        while (SystemClock.elapsedRealtime() < deadline) {
            find(predicate)?.let { return it }
            Thread.sleep(50)
        }
        error("Expected chat UI element was not exposed in the active window")
    }

    private fun awaitText(text: String) = awaitNode { it.text?.contains(text) == true }

    private fun click(node: AccessibilityNodeInfo) {
        var target = node
        while (!target.isClickable) target = requireNotNull(target.parent)
        check(target.performAction(AccessibilityNodeInfo.ACTION_CLICK))
    }

    fun run(): String = with(instrumentation) {
        val app = targetContext.applicationContext
        val activity = startActivitySync(Intent(app, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
        val message = mutableStateOf(Message(1, false, "", streaming = true))
        val tokens = mutableIntStateOf(0)
        val composer = mutableStateOf(false)
        val folder = File(app.filesDir, "bench").apply { mkdirs() }
        fun screenshot(name: String) {
            uiAutomation.waitForIdle(500, 5_000)
            val bitmap = requireNotNull(uiAutomation.takeScreenshot())
            File(folder, name).outputStream().use { check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) }
            bitmap.recycle()
        }
        try {
            runOnMainSync {
                activity.setContent {
                    EquityTheme {
                        Surface(Modifier.fillMaxSize().navigationBarsPadding()) {
                            if (composer.value) ChatScreen(EngineState(phase = Phase.Ready, modelName = "UI fixture"),
                                ChatsState(), onOpenModels = {}, onOpenChats = {})
                            else Box(Modifier.statusBarsPadding().padding(16.dp)) {
                                AssistantMessage(message.value, tokens.intValue)
                            }
                        }
                    }
                }
            }
            awaitText("Reading your message")
            runOnMainSync {
                message.value = message.value.copy(text = "\n".repeat(64) + "\u200b")
                tokens.intValue = 17
            }
            awaitText("Generating tokens; no visible text yet")
            awaitText("17 tokens")
            runOnMainSync { tokens.intValue = 18 }
            awaitText("18 tokens")
            screenshot("chat-blank-stream-ui.png")
            runOnMainSync { message.value = message.value.copy(text = "Hello from the UI fixture. Un caffè ☕.") }
            awaitText("Hello from the UI fixture.")
            check(find { it.text?.contains("no visible text yet") == true } == null)
            runOnMainSync { message.value = message.value.copy(text = "\n".repeat(64), streaming = false) }
            awaitText("The model finished without producing visible text.")

            val generation = Generation("ui-fixture", 1, SystemClock.elapsedRealtime(), tokens = 18)
            val notification = Notifications.progress(app, EngineState(phase = Phase.Generating, generation = generation))
            check(notification.extras.getCharSequence(Notification.EXTRA_TEXT).toString().contains("no visible text yet"))
            val visible = Notifications.progress(app, EngineState(phase = Phase.Generating,
                generation = generation.copy(hasVisibleText = true)))
            check(!visible.extras.getCharSequence(Notification.EXTRA_TEXT).toString().contains("no visible text yet"))

            runOnMainSync { composer.value = true }
            val configuration = awaitNode { it.contentDescription?.toString() == "Chat configuration" }
            val attachment = awaitNode { it.contentDescription?.toString() == "Attach" }
            val configurationBounds = Rect().also(configuration::getBoundsInScreen)
            val attachmentBounds = Rect().also(attachment::getBoundsInScreen)
            check(configurationBounds.left >= attachmentBounds.right)
            screenshot("chat-composer-ui.png")
            click(configuration)
            awaitText("Directional steering")
            awaitText("Signed strength")
            check(find { it.text?.toString() == "Adjust" } == null)
            screenshot("chat-configuration-ui.png")
            click(awaitText("Cancel"))
            File(folder, "chat-ui-validation.json").writeText(JSONObject()
                .put("passed", true).put("whitespace_stream_status_visible", true)
                .put("live_token_counter_updates", true).put("visible_unicode_stream_recomposes", true)
                .put("blank_completion_message", true).put("notification_matches_visibility", true)
                .put("configuration_next_to_attachment", true).put("single_configuration_modal", true)
                .put("model_inference_executed", false).put("chat_or_settings_modified", false).toString(2))
            "blank/visible streaming UI, live counter, notification state and composer configuration modal"
        } finally {
            runOnMainSync { activity.finish() }
        }
    }
}
