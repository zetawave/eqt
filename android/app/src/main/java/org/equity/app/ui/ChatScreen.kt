package org.equity.app.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.SuggestionChipDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import android.content.ClipData
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import org.equity.app.EngineController
import org.equity.app.EngineState
import org.equity.app.Message
import org.equity.app.Phase

private val Suggestions = listOf(
    "Spiegami in due frasi la differenza tra RAM e memoria di archiviazione.",
    "Write a short, friendly email to reschedule a meeting.",
    "Dammi tre idee per una cena veloce e sana.",
    "Explain how a mixture-of-experts model works, simply.",
)

@Composable
fun ChatScreen(engine: EngineState, onOpenModels: () -> Unit) {
    val context = LocalContext.current
    var input by rememberSaveable { mutableStateOf("") }
    val list = rememberLazyListState()
    val last = engine.messages.lastOrNull()
    LaunchedEffect(engine.messages.size, last?.text?.length) {
        if (engine.messages.isNotEmpty()) list.scrollToItem(engine.messages.lastIndex, Int.MAX_VALUE)
    }
    val send = { text: String ->
        EngineController.send(context, text)
        input = ""
    }
    Column(Modifier.fillMaxSize().imePadding()) {
        ChatHeader(engine)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (engine.messages.isEmpty()) {
                EmptyChat(engine, onOpenModels, onSuggestion = send)
            } else {
                LazyColumn(
                    state = list,
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                    modifier = Modifier.fillMaxSize(),
                ) {
                    items(engine.messages, key = { it.id }) { message ->
                        if (message.fromUser) UserBubble(message) else AssistantMessage(message)
                    }
                }
            }
        }
        Composer(
            value = input,
            onValueChange = { input = it },
            phase = engine.phase,
            onSend = { send(input) },
            onStop = { EngineController.stop() },
        )
    }
}

@Composable
private fun ChatHeader(engine: EngineState) {
    val colors = MaterialTheme.colorScheme
    Row(
        Modifier.fillMaxWidth().statusBarsPadding().padding(start = 16.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Logo(30.dp, wordmark = false)
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(engine.modelName ?: "Equity", style = MaterialTheme.typography.titleMedium, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusDot(engine.phase)
                Spacer(Modifier.width(6.dp))
                val detail = when (engine.phase) {
                    Phase.Empty -> "No model loaded"
                    Phase.Loading -> engine.loadPercent?.let { "Loading $it%" } ?: engine.status
                    Phase.Ready -> engine.lastSpeed?.let { "Ready · last %.1f tok/s".format(it) } ?: "Ready · offline"
                    else -> engine.status
                }
                Text(detail, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, maxLines = 1)
            }
        }
        IconButton(
            onClick = { EngineController.newChat() },
            enabled = engine.messages.isNotEmpty() && engine.phase != Phase.Generating,
        ) { Icon(Icons.Rounded.Add, contentDescription = "New chat") }
    }
    if (engine.phase == Phase.Loading) {
        val percent = engine.loadPercent
        if (percent != null) LinearProgressIndicator(progress = { percent / 100f }, modifier = Modifier.fillMaxWidth().height(2.dp))
        else LinearProgressIndicator(Modifier.fillMaxWidth().height(2.dp))
    }
}

@Composable
fun StatusDot(phase: Phase) {
    val colors = MaterialTheme.colorScheme
    val busy = phase == Phase.Loading || phase == Phase.Generating || phase == Phase.Unloading
    val pulse = rememberInfiniteTransition(label = "pulse")
    val alpha by pulse.animateFloat(1f, 0.35f, infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "alpha")
    val color = when (phase) {
        Phase.Ready -> colors.primary
        Phase.Empty -> colors.outline
        else -> colors.tertiary
    }
    Box(Modifier.size(8.dp).alpha(if (busy) alpha else 1f).background(color, CircleShape))
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun EmptyChat(engine: EngineState, onOpenModels: () -> Unit, onSuggestion: (String) -> Unit) {
    val colors = MaterialTheme.colorScheme
    Column(
        Modifier.fillMaxSize().padding(horizontal = 28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Logo(48.dp)
        Spacer(Modifier.height(16.dp))
        Text(
            "Private AI that runs on your phone.\nYour conversations never leave this device.",
            style = MaterialTheme.typography.bodyLarge, color = colors.onSurfaceVariant, textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(28.dp))
        when (engine.phase) {
            Phase.Empty -> Button(onClick = onOpenModels) { Text("Choose a model") }
            Phase.Loading, Phase.Unloading -> Text(engine.status, style = MaterialTheme.typography.bodyMedium, color = colors.onSurfaceVariant)
            else -> FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Suggestions.forEach { prompt ->
                    SuggestionChip(
                        onClick = { onSuggestion(prompt) },
                        label = { Text(prompt, maxLines = 2, style = MaterialTheme.typography.bodySmall) },
                        shape = RoundedCornerShape(16.dp),
                        colors = SuggestionChipDefaults.suggestionChipColors(containerColor = colors.surfaceContainer),
                        border = null,
                        modifier = Modifier.widthIn(max = 300.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun UserBubble(message: Message) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
        Surface(
            color = MaterialTheme.colorScheme.secondaryContainer,
            shape = RoundedCornerShape(20.dp, 20.dp, 6.dp, 20.dp),
            modifier = Modifier.widthIn(max = 320.dp),
        ) {
            SelectionContainer {
                Text(message.text, style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 11.dp))
            }
        }
    }
}

@Composable
private fun AssistantMessage(message: Message) {
    val colors = MaterialTheme.colorScheme
    val clipboard = LocalClipboard.current
    val scope = rememberCoroutineScope()
    Column(Modifier.fillMaxWidth().padding(end = 12.dp)) {
        if (message.streaming && message.text.isEmpty()) {
            TypingDots()
        } else {
            SelectionContainer {
                Text(
                    message.text + if (message.streaming) " ▍" else "",
                    style = MaterialTheme.typography.bodyLarge,
                    color = if (message.failed) colors.error else colors.onSurface,
                )
            }
        }
        if (!message.streaming) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                message.stats?.let {
                    Text(it, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                        modifier = Modifier.weight(1f, fill = false))
                }
                IconButton(onClick = { scope.launch { clipboard.setClipEntry(ClipEntry(ClipData.newPlainText("Equity answer", message.text))) } }, modifier = Modifier.size(36.dp)) {
                    Icon(Icons.Outlined.ContentCopy, contentDescription = "Copy answer", tint = colors.onSurfaceVariant,
                        modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

@Composable
private fun TypingDots() {
    val transition = rememberInfiniteTransition(label = "typing")
    Row(Modifier.padding(vertical = 10.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        repeat(3) { index ->
            val alpha by transition.animateFloat(
                initialValue = 0.25f, targetValue = 1f,
                animationSpec = infiniteRepeatable(tween(500, delayMillis = index * 160), RepeatMode.Reverse),
                label = "dot$index",
            )
            Box(Modifier.size(8.dp).alpha(alpha).background(MaterialTheme.colorScheme.primary, CircleShape))
        }
    }
}

@Composable
private fun Composer(value: String, onValueChange: (String) -> Unit, phase: Phase, onSend: () -> Unit, onStop: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    val ready = phase == Phase.Ready
    val generating = phase == Phase.Generating
    Column(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 6.dp, bottom = 6.dp)) {
        Surface(color = colors.surfaceContainerHigh, shape = RoundedCornerShape(28.dp)) {
            Row(verticalAlignment = Alignment.Bottom, modifier = Modifier.padding(end = 6.dp, bottom = 6.dp)) {
                TextField(
                    value = value,
                    onValueChange = onValueChange,
                    enabled = ready || generating,
                    placeholder = { Text(if (phase == Phase.Empty) "Load a model to start" else "Message Equity") },
                    maxLines = 6,
                    textStyle = MaterialTheme.typography.bodyLarge,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color.Transparent,
                        unfocusedContainerColor = Color.Transparent,
                        disabledContainerColor = Color.Transparent,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        disabledIndicatorColor = Color.Transparent,
                    ),
                    modifier = Modifier.weight(1f),
                )
                if (generating) {
                    FilledIconButton(
                        onClick = onStop,
                        colors = IconButtonDefaults.filledIconButtonColors(containerColor = colors.onSurface, contentColor = colors.surface),
                    ) { Icon(Icons.Rounded.Stop, contentDescription = "Stop") }
                } else {
                    FilledIconButton(onClick = onSend, enabled = ready && value.isNotBlank()) {
                        Icon(Icons.AutoMirrored.Rounded.Send, contentDescription = "Send")
                    }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Lock, contentDescription = null, tint = colors.onSurfaceVariant, modifier = Modifier.size(12.dp))
            Spacer(Modifier.width(4.dp))
            Text("Runs on this device. Nothing is sent online.", style = MaterialTheme.typography.labelSmall,
                color = colors.onSurfaceVariant)
        }
    }
}
