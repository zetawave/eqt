package org.equity.app.ui

import androidx.compose.ui.platform.LocalContext
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
import org.equity.app.R
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
    val strings = LocalContext.current.resources
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
    AlertDialog(onDismissRequest = onDismiss, title = {
        Text(strings.getString(R.string.chat_settings), style = MaterialTheme.typography.titleLarge)
    }, text = {
        Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(strings.getString(R.string.saved_for_this_chat_changes_apply_to_the_next_answer), style = small, color = muted)
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                HelpLabel(
                    strings.getString(R.string.remember_this_chat),
                    strings.getString(R.string.after_each_answer_the_model_s_state_of_this_conversation_is_saved_on_this_p),
                    summary = strings.getString(R.string.remember_chat_summary),
                    modifier = Modifier.weight(1f), style = MaterialTheme.typography.titleSmall,
                )
                Switch(checked = session, onCheckedChange = { session = it }, enabled = !engine.busy)
            }
            Text(
                when {
                    saved != null -> strings.getString(R.string.saved_1_s_2_d_tokens, formatBytes(saved!!.bytes), saved!!.tokens)
                    session -> strings.getString(R.string.nothing_saved_yet)
                    else -> strings.getString(R.string.off)
                } + if (thinking && session) strings.getString(R.string.not_used_while_think_is_on) else "",
                style = small, color = muted,
            )
            if (saved != null) TextButton(onClick = {
                chatId?.let(ChatStore::deleteSession)
                refresh++
            }, enabled = !engine.busy) { Text(strings.getString(R.string.delete_saved_state)) }
            if (available) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(strings.getString(R.string.directional_steering), style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                    Switch(checked = enabled, onCheckedChange = { enabled = it }, enabled = !engine.busy)
                }
                SliderField(
                    label = strings.getString(R.string.signed_strength),
                    value = strength,
                    onValueChange = { strength = it },
                    range = SteeringOptions.range,
                    step = 0.05f,
                    format = { "%.2f".format(Locale.US, it) },
                    enabled = !engine.busy,
                    accept = { it != 0f },
                    hint = strings.getString(R.string.nonzero_from_1_to_2) + "\n\n" +
                        strings.getString(R.string.negative_values_remove_the_vector_s_direction_from_every_layer_1_removes_it),
                    summary = strings.getString(R.string.steering_strength_summary),
                    validationHint = strings.getString(R.string.nonzero_from_1_to_2),
                )
            }
            if (engine.busy) Text(strings.getString(R.string.wait_for_the_current_operation_to_finish_before_changing_these_settings), style = small)
        }
    }, confirmButton = {
        TextButton(enabled = !engine.busy && valid, onClick = {
            if (session != chats.session) ChatStore.setSession(session)
            if (available) ChatStore.setSteering(SteeringOptions(enabled, strength))
            onDismiss()
        }) { Text(strings.getString(R.string.save)) }
    }, dismissButton = { TextButton(onClick = onDismiss) { Text(strings.getString(R.string.cancel)) } },
        containerColor = MaterialTheme.colorScheme.surfaceContainerHigh)
}
