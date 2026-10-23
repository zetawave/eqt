package org.equity.app.ui

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.outlined.AttachFile
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Psychology
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.ExpandLess
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.SuggestionChipDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.equity.app.R
import org.equity.app.Attachment
import org.equity.app.AttachmentException
import org.equity.app.AttachmentKind
import org.equity.app.Attachments
import org.equity.app.ChatStore
import org.equity.app.ChatsState
import org.equity.app.EngineController
import org.equity.app.EngineState
import org.equity.app.Message
import org.equity.app.Phase
import org.equity.app.SettingsStore
import org.equity.app.hasVisibleContent
import java.util.Locale
import java.io.File
import kotlin.math.roundToInt

private val Suggestions = listOf(
    R.string.explain_the_difference_between_ram_and_storage_in_two_sentences,
    R.string.write_a_short_friendly_email_to_reschedule_a_meeting,
    R.string.give_me_three_ideas_for_a_quick_healthy_dinner,
    R.string.explain_how_a_mixture_of_experts_model_works_simply,
)

// Matches the number of images the engine is shown per request.
private const val MAX_IMAGES_PER_MESSAGE = 4

@Composable
fun ChatScreen(engine: EngineState, chats: ChatsState, onOpenModels: () -> Unit, onOpenChats: () -> Unit) {
    val strings = LocalContext.current.resources
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var input by rememberSaveable { mutableStateOf("") }
    val pending = remember { mutableStateListOf<Attachment>() }
    var importing by remember { mutableIntStateOf(0) }
    var viewing by remember { mutableStateOf<File?>(null) }
    var configuring by remember { mutableStateOf(false) }
    val settings by SettingsStore.state.collectAsState()
    val vision by rememberUpdatedState(engine.vision)
    val chat = chats.current
    val messages = chat?.messages.orEmpty()
    val list = rememberLazyListState()
    val last = messages.lastOrNull()
    LaunchedEffect(chat?.id, messages.size, last?.text?.length) {
        if (messages.isNotEmpty()) list.scrollToItem(messages.lastIndex, Int.MAX_VALUE)
    }

    val import = { uris: List<Uri>, cleanup: (() -> Unit)? ->
        scope.launch {
            importing++
            for (uri in uris) {
                if (pending.count { it.kind == AttachmentKind.Image } >= MAX_IMAGES_PER_MESSAGE) {
                    Toast.makeText(context, strings.getString(R.string.up_to_1_d_images_per_message, MAX_IMAGES_PER_MESSAGE), Toast.LENGTH_SHORT).show()
                    break
                }
                try {
                    pending += withContext(Dispatchers.IO) { Attachments.import(context, uri, vision) }
                } catch (error: AttachmentException) {
                    Toast.makeText(context, error.message, Toast.LENGTH_LONG).show()
                } catch (error: Exception) {
                    Toast.makeText(context, strings.getString(R.string.cannot_attach_this_file), Toast.LENGTH_LONG).show()
                }
            }
            cleanup?.invoke()
            importing--
        }
    }
    val photos = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(MAX_IMAGES_PER_MESSAGE)) {
        if (it.isNotEmpty()) import(it, null)
    }
    val documents = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) {
        if (it.isNotEmpty()) import(it, null)
    }
    var capture by rememberSaveable { mutableStateOf<String?>(null) }
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { taken ->
        val file = capture?.let(::File)
        capture = null
        if (file != null) {
            // The provider URI reports image/jpeg; a file:// URI reports no type at all.
            if (taken && file.length() > 0) {
                import(listOf(FileProvider.getUriForFile(context, "${context.packageName}.files", file))) { file.delete() }
            } else {
                file.delete()
            }
        }
    }
    val takePhoto: () -> Unit = {
        val file = File(context.cacheDir, "camera/${System.currentTimeMillis()}.jpg").apply { parentFile?.mkdirs() }
        capture = file.path
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
        try {
            camera.launch(uri)
        } catch (_: ActivityNotFoundException) {
            capture = null
            Toast.makeText(context, strings.getString(R.string.no_camera_app_is_available), Toast.LENGTH_SHORT).show()
        }
    }

    var askedNotifications by rememberSaveable { mutableStateOf(false) }
    val notificationPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    val send = { text: String ->
        if (EngineController.send(context, chats.currentId, text, pending.toList())) {
            input = ""
            pending.clear()
            // The finished-answer notification needs this permission on Android 13+; asked once, on first use.
            val missing = Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(context,
                Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
            if (missing && !askedNotifications) {
                askedNotifications = true
                notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }
    val answeringElsewhere = engine.generation != null && engine.generation.chatId != chats.currentId
    // Context this chat fills: what its last finished answer left in use, plus the tokens arriving for a new answer
    // (whose new prompt text is counted once it finishes).
    val contextUsed = (messages.lastOrNull { !it.fromUser && it.contextTokens != null }?.contextTokens ?: 0) +
        (engine.generation?.takeIf { it.chatId == chats.currentId }?.tokens ?: 0)
    Column(Modifier.fillMaxSize().imePadding()) {
        ChatHeader(engine, chat?.title, onOpenChats, onNewChat = { ChatStore.newChat() }, canStartNew = chat != null)
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (messages.isEmpty()) {
                EmptyChat(engine, onOpenModels, onSuggestion = send)
            } else {
                LazyColumn(
                    state = list,
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                    modifier = Modifier.fillMaxSize(),
                ) {
                    items(messages, key = { it.id }) { message ->
                        if (message.fromUser) UserMessage(chat!!.id, message, onView = { viewing = it })
                        else AssistantMessage(message, engine.generation?.takeIf {
                            it.chatId == chats.currentId && it.answerId == message.id
                        }?.tokens)
                    }
                }
            }
        }
        Composer(
            value = input,
            onValueChange = { input = it },
            engine = engine,
            answeringElsewhere = answeringElsewhere,
            pending = pending,
            importing = importing > 0,
            onRemove = { attachment ->
                pending.remove(attachment)
                Attachments.discard(attachment)
            },
            onPhotos = { photos.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            onCamera = takePhoto,
            onFile = { documents.launch(arrayOf("*/*")) },
            onConfigure = { configuring = true },
            configured = chats.session || (chats.steering.enabled && engine.steeringProfile != null),
            thinking = settings.thinking && engine.thinkingSupported,
            thinkingSupported = engine.thinkingSupported,
            onToggleThinking = { SettingsStore.update(context) { it.copy(thinking = !it.thinking) } },
            onView = { viewing = it },
            onSend = { send(input) },
            onStop = { EngineController.stop() },
            contextUsed = contextUsed,
        )
    }
    viewing?.let { ImageViewer(it) { viewing = null } }
    if (configuring) ChatConfigurationDialog(engine, chats, settings.thinking && engine.thinkingSupported) { configuring = false }
}

@Composable
private fun ChatHeader(engine: EngineState, title: String?, onOpenChats: () -> Unit, onNewChat: () -> Unit, canStartNew: Boolean) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    Row(
        Modifier.fillMaxWidth().statusBarsPadding().padding(start = 4.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onOpenChats) { Icon(Icons.Rounded.Menu, contentDescription = strings.getString(R.string.chats)) }
        Spacer(Modifier.width(4.dp))
        Column(Modifier.weight(1f)) {
            Text(title ?: if (engine.phase == Phase.Empty) "Equity" else strings.getString(R.string.new_chat), style = MaterialTheme.typography.titleMedium, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusDot(engine.phase)
                Spacer(Modifier.width(6.dp))
                val detail = when (engine.phase) {
                    Phase.Empty -> strings.getString(R.string.no_model_loaded)
                    Phase.Loading -> engine.loadPercent?.let { strings.getString(R.string.loading_1_d, it) } ?: engine.status
                    Phase.Ready -> buildString {
                        append(engine.modelName ?: strings.getString(R.string.ready))
                        engine.lastSpeed?.let { append(strings.getString(R.string.last_1_1f_tok_s, it)) }
                    }
                    else -> engine.status
                }
                Text(detail, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, maxLines = 1,
                    overflow = TextOverflow.Ellipsis)
            }
        }
        IconButton(onClick = onNewChat, enabled = canStartNew) { Icon(Icons.Rounded.Add, contentDescription = strings.getString(R.string.new_chat)) }
    }
    if (engine.phase == Phase.Loading) {
        val percent = engine.loadPercent
        GradientProgress(percent?.let { it / 100f })
    } else {
        // Hairline under the header; keeps the chat surface visually anchored while scrolling.
        Box(
            Modifier.fillMaxWidth().height(1.dp)
                .background(Brush.horizontalGradient(listOf(Color.Transparent, colors.outlineVariant, Color.Transparent))),
        )
    }
}

@Composable
fun StatusDot(phase: Phase) {
    val colors = MaterialTheme.colorScheme
    val color = when (phase) {
        Phase.Ready -> colors.primary
        Phase.Empty -> colors.outline
        else -> colors.tertiary
    }
    // Only loading pulses. Any running animation redraws the screen every frame, and while an answer is written
    // each frame takes a big core from the compute threads.
    val alpha = if (phase == Phase.Loading || phase == Phase.Unloading) {
        val pulse = rememberInfiniteTransition(label = "pulse")
        pulse.animateFloat(1f, 0.35f, infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "alpha").value
    } else 1f
    Box(Modifier.size(8.dp).alpha(alpha).background(color, CircleShape))
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun EmptyChat(engine: EngineState, onOpenModels: () -> Unit, onSuggestion: (String) -> Unit) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    // A single fade-and-rise on entry; nothing animates after that, so streaming answers keep the big cores.
    var shown by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { shown = true }
    val alpha by animateFloatAsState(if (shown) 1f else 0f, tween(500), label = "emptyAlpha")
    val rise by animateFloatAsState(if (shown) 0f else 28f, tween(500), label = "emptyRise")
    Column(
        Modifier.fillMaxSize().padding(horizontal = 28.dp).graphicsLayer(alpha = alpha, translationY = rise),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Logo(52.dp)
        Spacer(Modifier.height(36.dp))
        when (engine.phase) {
            Phase.Empty -> Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                GradientButton(onClick = onOpenModels, icon = Icons.AutoMirrored.Filled.ArrowForward) {
                    Text(strings.getString(R.string.choose_a_model), style = MaterialTheme.typography.labelLarge)
                }
            }
            Phase.Loading, Phase.Unloading -> EmptyStatus(engine.status)
            Phase.Generating -> EmptyStatus(strings.getString(R.string.equity_is_answering_in_another_chat))
            Phase.Ready -> FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Suggestions.forEach { promptId ->
                    val prompt = strings.getString(promptId)
                    SuggestionChip(
                        onClick = { onSuggestion(prompt) },
                        label = { Text(prompt, maxLines = 2, style = MaterialTheme.typography.bodySmall) },
                        shape = RoundedCornerShape(16.dp),
                        colors = SuggestionChipDefaults.suggestionChipColors(containerColor = colors.surfaceContainer.copy(alpha = 0.6f)),
                        border = BorderStroke(1.dp, colors.outlineVariant),
                        modifier = Modifier.widthIn(max = 300.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyStatus(text: String) {
    Text(text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        textAlign = TextAlign.Center)
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun UserMessage(chatId: String, message: Message, onView: (File) -> Unit) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
        if (message.attachments.isNotEmpty()) {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.End),
                verticalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.widthIn(max = 320.dp).padding(bottom = 6.dp),
            ) {
                message.attachments.forEach { attachment ->
                    val file = ChatStore.file(chatId, attachment)
                    if (attachment.kind == AttachmentKind.Image) ImageThumbnail(file, 120.dp) { onView(file) }
                    else DocumentChip(attachment)
                }
            }
        }
        if (message.text.isNotEmpty()) {
            val bubbleShape = RoundedCornerShape(20.dp, 20.dp, 6.dp, 20.dp)
            Surface(
                color = MaterialTheme.colorScheme.secondaryContainer,
                shape = bubbleShape,
                modifier = Modifier.widthIn(max = 320.dp)
                    .border(BorderStroke(1.dp, Brush.linearGradient(listOf(Teal.copy(alpha = 0.45f), Violet.copy(alpha = 0.45f)))), bubbleShape),
            ) {
                SelectionContainer {
                    Text(message.text, style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 11.dp))
                }
            }
        }
    }
}

@Composable
internal fun AssistantMessage(message: Message, generatedTokens: Int? = null) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    val clipboard = LocalClipboard.current
    val scope = rememberCoroutineScope()
    Column(Modifier.fillMaxWidth().padding(end = 12.dp)) {
        // The closing tag records the reasoning time, so a streaming answer without one is still reasoning.
        val reasoning = message.reasoning
        val reasoningNow = reasoning != null && message.streaming && message.reasoningMs == null
        if (reasoning != null) ReasoningBlock(reasoning, reasoningNow, message.reasoningMs, generatedTokens)
        if (reasoningNow) {
            // The reasoning block is the progress indicator.
        } else if (!message.text.hasVisibleContent()) {
            // Static: an animation here would compete with the prompt's evaluation for the big cores.
            val sampled = (generatedTokens ?: 0) > 0 || message.text.isNotEmpty()
            Text(when {
                !message.streaming && reasoning != null -> strings.getString(R.string.the_model_stopped_while_reasoning_before_writing_an_answer)
                !message.streaming -> strings.getString(R.string.the_model_finished_without_producing_visible_text)
                sampled -> strings.getString(R.string.generating_tokens_no_visible_text_yet)
                else -> strings.getString(R.string.reading_your_message)
            }, style = MaterialTheme.typography.bodyMedium, color = colors.onSurfaceVariant,
                modifier = Modifier.padding(vertical = 8.dp))
            if (message.streaming && generatedTokens != null && generatedTokens > 0) {
                Text(strings.getString(R.string.value_1_d_tokens, generatedTokens), style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant)
            }
        } else {
            SelectionContainer {
                MarkdownText(
                    message.text + if (message.streaming) " ▍" else "",
                    color = if (message.failed) colors.error else colors.onSurface,
                )
            }
        }
        if (!message.streaming) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                val note = message.stats ?: if (message.failed) strings.getString(R.string.not_completed) else null
                note?.let {
                    Text(it, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                        modifier = Modifier.weight(1f, fill = false))
                }
                IconButton(onClick = { scope.launch { clipboard.setClipEntry(ClipEntry(ClipData.newPlainText(strings.getString(R.string.equity_answer), message.text))) } }, modifier = Modifier.size(36.dp)) {
                    Icon(Icons.Outlined.ContentCopy, contentDescription = strings.getString(R.string.copy_answer), tint = colors.onSurfaceVariant,
                        modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

/**
 * The model's reasoning above its answer: collapsed to a header, with only the newest lines fading in while it is
 * written; tapping the header shows all of it.
 */
@Composable
private fun ReasoningBlock(reasoning: String, inProgress: Boolean, reasoningMs: Long?, generatedTokens: Int?) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    var expanded by rememberSaveable { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(bottom = 6.dp)) {
        Surface(
            color = colors.surfaceContainerLow.copy(alpha = 0.6f),
            shape = RoundedCornerShape(14.dp),
            border = BorderStroke(1.dp, colors.outlineVariant),
        ) {
            Row(
                Modifier.clip(RoundedCornerShape(14.dp)).clickable { expanded = !expanded }.padding(vertical = 6.dp, horizontal = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
            Icon(Icons.Outlined.Psychology, contentDescription = null, tint = colors.onSurfaceVariant, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text(when {
                inProgress -> strings.getString(R.string.thinking) + (generatedTokens?.takeIf { it > 0 }?.let { strings.getString(R.string.value_1_d_tokens_190, it) } ?: "")
                reasoningMs != null -> strings.getString(R.string.thought_for_1_1f_s, reasoningMs / 1000.0)
                else -> strings.getString(R.string.reasoning)
            }, style = MaterialTheme.typography.labelLarge, color = colors.onSurfaceVariant)
            Icon(if (expanded) Icons.Rounded.ExpandLess else Icons.Rounded.ExpandMore,
                contentDescription = if (expanded) strings.getString(R.string.hide_reasoning) else strings.getString(R.string.show_reasoning),
                tint = colors.onSurfaceVariant, modifier = Modifier.size(18.dp))
            }
        }
        when {
            expanded -> Row(Modifier.height(IntrinsicSize.Min).padding(start = 12.dp, top = 6.dp)) {
                Box(Modifier.width(2.dp).fillMaxHeight()
                    .background(Brush.verticalGradient(listOf(Teal.copy(alpha = 0.7f), Violet.copy(alpha = 0.7f)))))
                SelectionContainer {
                    Text(reasoning.ifEmpty { "…" }, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
                        modifier = Modifier.padding(start = 10.dp))
                }
            }
            inProgress && reasoning.isNotEmpty() -> ReasoningPreview(reasoning)
        }
    }
}

/** The newest reasoning lines, bottom-aligned and fading upward, so the line being written stays readable. */
@Composable
private fun ReasoningPreview(reasoning: String) {
    // Only the tail is laid out: enough characters to fill the preview at any phone width.
    val tail = reasoning.takeLast(480)
    Box(
        Modifier.fillMaxWidth().height(52.dp).padding(start = 16.dp).clipToBounds()
            .graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)
            .drawWithContent {
                drawContent()
                drawRect(Brush.verticalGradient(0f to Color.Transparent, 0.7f to Color.Black), blendMode = BlendMode.DstIn)
            },
    ) {
        Text(tail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.align(Alignment.BottomStart).wrapContentHeight(Alignment.Bottom, unbounded = true))
    }
}

@Composable
private fun Composer(
    value: String,
    onValueChange: (String) -> Unit,
    engine: EngineState,
    answeringElsewhere: Boolean,
    pending: List<Attachment>,
    importing: Boolean,
    onRemove: (Attachment) -> Unit,
    onPhotos: () -> Unit,
    onCamera: () -> Unit,
    onFile: () -> Unit,
    onConfigure: () -> Unit,
    configured: Boolean,
    thinking: Boolean,
    thinkingSupported: Boolean,
    onToggleThinking: () -> Unit,
    onView: (File) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    contextUsed: Int,
) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    val ready = engine.phase == Phase.Ready
    val generating = engine.phase == Phase.Generating
    var menu by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 6.dp, bottom = 6.dp)) {
        Surface(
            color = colors.surfaceContainer.copy(alpha = 0.85f),
            shape = RoundedCornerShape(28.dp),
            border = BorderStroke(1.dp, colors.outlineVariant),
        ) {
            Column {
                if (pending.isNotEmpty() || importing) {
                    Row(
                        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(start = 12.dp, end = 12.dp, top = 12.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        pending.forEach { attachment ->
                            if (attachment.kind == AttachmentKind.Image) {
                                val file = File(ChatStore.staging, attachment.file)
                                Box {
                                    ImageThumbnail(file, 64.dp) { onView(file) }
                                    RemoveBadge(Modifier.align(Alignment.TopEnd)) { onRemove(attachment) }
                                }
                            } else {
                                DocumentChip(attachment) {
                                    IconButton(onClick = { onRemove(attachment) }, modifier = Modifier.size(28.dp)) {
                                        Icon(Icons.Rounded.Close, contentDescription = strings.getString(R.string.remove_1_s, attachment.name), modifier = Modifier.size(16.dp))
                                    }
                                }
                            }
                        }
                        if (importing) CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
                    }
                }
                TextField(
                    value = value,
                    onValueChange = onValueChange,
                    enabled = ready || generating,
                    placeholder = {
                        Text(when {
                            engine.phase == Phase.Empty -> strings.getString(R.string.load_a_model_to_start)
                            answeringElsewhere -> strings.getString(R.string.equity_is_answering_in_another_chat_23)
                            else -> strings.getString(R.string.message_equity)
                        })
                    },
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
                    modifier = Modifier.fillMaxWidth().padding(start = 4.dp, end = 4.dp),
                )
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 4.dp, end = 6.dp, bottom = 6.dp)) {
                    Box {
                        IconButton(onClick = { menu = true }, enabled = ready || generating) {
                            Icon(Icons.Outlined.AttachFile, contentDescription = strings.getString(R.string.attach))
                        }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }, containerColor = colors.surfaceContainerHighest) {
                            val imageHint = if (engine.vision) null else strings.getString(R.string.needs_a_model_with_image_support)
                            DropdownMenuItem(
                                text = { MenuLabel(strings.getString(R.string.photos), imageHint) },
                                leadingIcon = { Icon(Icons.Outlined.PhotoLibrary, contentDescription = null) },
                                enabled = engine.vision,
                                onClick = { menu = false; onPhotos() },
                            )
                            DropdownMenuItem(
                                text = { MenuLabel(strings.getString(R.string.camera), imageHint) },
                                leadingIcon = { Icon(Icons.Outlined.PhotoCamera, contentDescription = null) },
                                enabled = engine.vision,
                                onClick = { menu = false; onCamera() },
                            )
                            DropdownMenuItem(
                                text = { MenuLabel(strings.getString(R.string.file), strings.getString(R.string.text_code_or_pdf)) },
                                leadingIcon = { Icon(Icons.Outlined.Description, contentDescription = null) },
                                onClick = { menu = false; onFile() },
                            )
                        }
                    }
                    // Chat settings: the saved state, and steering while the loaded model has an imported vector.
                    IconButton(onClick = onConfigure) {
                        Icon(Icons.Outlined.Tune, contentDescription = strings.getString(R.string.chat_settings),
                            tint = if (configured) colors.primary else colors.onSurfaceVariant)
                    }
                    // Applies to the next message; disabled when the loaded model's template has no thinking mode.
                    FilterChip(
                        selected = thinking,
                        onClick = onToggleThinking,
                        enabled = thinkingSupported,
                        label = { Text(strings.getString(R.string.think)) },
                        leadingIcon = { Icon(Icons.Outlined.Psychology, contentDescription = null, modifier = Modifier.size(18.dp)) },
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.padding(start = 2.dp),
                    )
                    Spacer(Modifier.weight(1f))
                    engine.contextSize?.takeIf { ready || generating }?.let { size -> ContextGauge(contextUsed, size) }
                    if (generating) {
                        FilledIconButton(
                            onClick = onStop,
                            colors = IconButtonDefaults.filledIconButtonColors(containerColor = colors.onSurface, contentColor = colors.surface),
                        ) { Icon(Icons.Rounded.Stop, contentDescription = strings.getString(R.string.stop)) }
                    } else {
                        val sendEnabled = ready && !importing && (value.isNotBlank() || pending.isNotEmpty())
                        Box(
                            Modifier.size(44.dp).clip(CircleShape)
                                .alpha(if (sendEnabled) 1f else 0.4f)
                                .background(BrandGradient)
                                .clickable(enabled = sendEnabled, onClick = onSend),
                            contentAlignment = Alignment.Center,
                        ) {
                            Icon(Icons.AutoMirrored.Rounded.Send, contentDescription = strings.getString(R.string.send),
                                tint = colors.onPrimary, modifier = Modifier.size(20.dp))
                        }
                    }
                }
            }
        }
    }
}

/** How much of the loaded context window this chat uses; a tap shows the numbers. Redrawn only as tokens arrive. */
@Composable
private fun ContextGauge(used: Int, size: Int) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    val share = (used.toFloat() / size).coerceIn(0f, 1f)
    var open by remember { mutableStateOf(false) }
    Box {
        IconButton(onClick = { open = true }, modifier = Modifier.size(40.dp)) {
            GradientGauge(share = share, warn = share >= 0.9f)
        }
        if (open) {
            HelpDialog(strings.getString(R.string.context), { open = false }) {
                Text(strings.getString(R.string.value_1_d_of_2_d_tokens_3_d, used, size, (share * 100).roundToInt()),
                    style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                Text(if (share >= 0.9f) strings.getString(R.string.nearly_full_a_new_chat_starts_fresh) else strings.getString(R.string.grows_with_the_conversation),
                    style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun MenuLabel(title: String, hint: String?) {
    Column {
        Text(title)
        hint?.let { Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
    }
}

@Composable
private fun RemoveBadge(modifier: Modifier, onClick: () -> Unit) {
    val strings = LocalContext.current.resources
    Surface(color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f), shape = CircleShape,
        modifier = modifier.padding(3.dp).size(22.dp), onClick = onClick) {
        Box(contentAlignment = Alignment.Center) {
            Icon(Icons.Rounded.Close, contentDescription = strings.getString(R.string.remove_image), modifier = Modifier.size(14.dp))
        }
    }
}
