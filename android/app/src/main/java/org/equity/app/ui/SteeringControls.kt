package org.equity.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.equity.app.ChatStore
import org.equity.app.EngineState
import org.equity.app.SteeringOptions
import java.util.Locale

/** Per-chat steering; chats open it only while the loaded model has an imported vector. */
@Composable
fun ChatConfigurationDialog(engine: EngineState, options: SteeringOptions, onDismiss: () -> Unit) {
    var enabled by remember(options) { mutableStateOf(options.enabled) }
    var strength by remember(options) { mutableFloatStateOf(options.strength) }
    val available = engine.steeringProfile != null
    val valid = SteeringOptions.valid(strength)
    AlertDialog(onDismissRequest = onDismiss, title = { Text("Chat configuration") }, text = {
        Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("Directional steering", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                Switch(checked = enabled, onCheckedChange = { enabled = it }, enabled = !engine.busy && available)
            }
            SliderField(
                label = "Signed strength",
                value = strength,
                onValueChange = { strength = it },
                range = SteeringOptions.range,
                step = 0.05f,
                format = { "%.2f".format(Locale.US, it) },
                enabled = available && !engine.busy,
                accept = { it != 0f },
                hint = "Nonzero, from -1 to 2.",
            )
            Text("Negative values remove the vector's direction from every layer; -1 removes it entirely. " +
                "Positive values add the direction at its layer. Strength is an experimental manual setting.",
                style = MaterialTheme.typography.bodySmall)
            Text(if (engine.busy) "Wait for the current operation to finish before changing these settings."
                else "Saved for this chat. Changes apply to the next answer.", style = MaterialTheme.typography.bodySmall)
        }
    }, confirmButton = {
        TextButton(enabled = !engine.busy && valid && available, onClick = {
            ChatStore.setSteering(SteeringOptions(enabled, strength))
            onDismiss()
        }) { Text("Save") }
    }, dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } })
}
