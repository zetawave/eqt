package org.equity.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.equity.app.ChatStore
import org.equity.app.ChatsState
import org.equity.app.EngineState
import org.equity.app.SessionSummary
import org.equity.app.SteeringOptions
import org.equity.app.formatBytes
import java.util.Locale

/**
 * Per-chat settings: keeping the model's state of the chat on storage, and directional steering while the loaded
 * model has an imported vector.
 */
@Composable
fun ChatConfigurationDialog(engine: EngineState, chats: ChatsState, thinking: Boolean, onDismiss: () -> Unit) {
    val options = chats.steering
    var enabled by remember(options) { mutableStateOf(options.enabled) }
    var strength by remember(options) { mutableFloatStateOf(options.strength) }
    var session by remember(chats.session) { mutableStateOf(chats.session) }
    var refresh by remember { mutableIntStateOf(0) }
    val chatId = chats.currentId
    val saved by produceState<SessionSummary?>(null, chatId, refresh) {
        value = chatId?.let { id -> withContext(Dispatchers.IO) { ChatStore.sessionSummary(id) } }
    }
    val available = engine.steeringProfile != null
    val valid = !available || SteeringOptions.valid(strength)
    val small = MaterialTheme.typography.bodySmall
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    AlertDialog(onDismissRequest = onDismiss, title = { Text("Chat settings") }, text = {
        Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("Remember this chat", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                Switch(checked = session, onCheckedChange = { session = it }, enabled = !engine.busy)
            }
            Text("After each answer the model's state of this conversation is saved on this phone. Coming back to the " +
                "chat, after another chat or a restart, continues at once instead of reading the whole conversation " +
                "again. It uses storage, grows with the chat and is deleted with it. A state is saved only when the " +
                "model's next turn can continue from it; some models' chat templates never allow that.",
                style = small, color = muted)
            Text(
                when {
                    saved != null -> "Saved: ${formatBytes(saved!!.bytes)} · %,d tokens".format(Locale.US, saved!!.tokens)
                    session -> "Nothing saved yet."
                    else -> "Off."
                } + if (thinking && session) " Not used while Think is on." else "",
                style = small, color = muted,
            )
            if (saved != null) TextButton(onClick = {
                chatId?.let(ChatStore::deleteSession)
                refresh++
            }, enabled = !engine.busy) { Text("Delete saved state") }
            if (available) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("Directional steering", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                    Switch(checked = enabled, onCheckedChange = { enabled = it }, enabled = !engine.busy)
                }
                SliderField(
                    label = "Signed strength",
                    value = strength,
                    onValueChange = { strength = it },
                    range = SteeringOptions.range,
                    step = 0.05f,
                    format = { "%.2f".format(Locale.US, it) },
                    enabled = !engine.busy,
                    accept = { it != 0f },
                    hint = "Nonzero, from -1 to 2.",
                )
                Text("Negative values remove the vector's direction from every layer; -1 removes it entirely. " +
                    "Positive values add the direction at its layer. Strength is an experimental manual setting. " +
                    "A steered answer is not saved as the chat's state.", style = small, color = muted)
            }
            Text(if (engine.busy) "Wait for the current operation to finish before changing these settings."
                else "Saved for this chat. Changes apply to the next answer.", style = small)
        }
    }, confirmButton = {
        TextButton(enabled = !engine.busy && valid, onClick = {
            if (session != chats.session) ChatStore.setSession(session)
            if (available) ChatStore.setSteering(SteeringOptions(enabled, strength))
            onDismiss()
        }) { Text("Save") }
    }, dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } })
}
