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
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.outlined.AttachFile
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.equity.app.Attachment
import org.equity.app.AttachmentException
import org.equity.app.AttachmentKind
import org.equity.app.Attachments
import org.equity.app.Catalog
import org.equity.app.ChatStore
import org.equity.app.ChatsState
import org.equity.app.EngineController
import org.equity.app.EngineState
import org.equity.app.Message
import org.equity.app.Phase
import org.equity.app.hasVisibleContent
import java.io.File

private val Suggestions = listOf(
    "Spiegami in due frasi la differenza tra RAM e memoria di archiviazione.",
    "Write a short, friendly email to reschedule a meeting.",
    "Dammi tre idee per una cena veloce e sana.",
    "Explain how a mixture-of-experts model works, simply.",
)

// Matches the number of images the engine is shown per request.
private const val MAX_IMAGES_PER_MESSAGE = 4

@Composable
fun ChatScreen(engine: EngineState, chats: ChatsState, onOpenModels: () -> Unit, onOpenChats: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var input by rememberSaveable { mutableStateOf("") }
    val pending = remember { mutableStateListOf<Attachment>() }
    var importing by remember { mutableIntStateOf(0) }
    var viewing by remember { mutableStateOf<File?>(null) }
    var configuring by remember { mutableStateOf(false) }
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
                    Toast.makeText(context, "Up to $MAX_IMAGES_PER_MESSAGE images per message", Toast.LENGTH_SHORT).show()
                    break
                }
                try {
                    pending += withContext(Dispatchers.IO) { Attachments.import(context, uri, vision) }
                } catch (error: AttachmentException) {
                    Toast.makeText(context, error.message, Toast.LENGTH_LONG).show()
                } catch (error: Exception) {
                    Toast.makeText(context, "Cannot attach this file", Toast.LENGTH_LONG).show()
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
            Toast.makeText(context, "No camera app is available", Toast.LENGTH_SHORT).show()
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
            steeringEnabled = chats.steering.enabled,
            onView = { viewing = it },
            onSend = { send(input) },
            onStop = { EngineController.stop() },
        )
    }
    viewing?.let { ImageViewer(it) { viewing = null } }
    if (configuring) ChatConfigurationDialog(engine, chats.steering) { configuring = false }
}

@Composable
private fun ChatHeader(engine: EngineState, title: String?, onOpenChats: () -> Unit, onNewChat: () -> Unit, canStartNew: Boolean) {
    val colors = MaterialTheme.colorScheme
    val context = LocalContext.current
    Row(
        Modifier.fillMaxWidth().statusBarsPadding().padding(start = 4.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onOpenChats) { Icon(Icons.Rounded.Menu, contentDescription = "Chats") }
        Spacer(Modifier.width(4.dp))
        Column(Modifier.weight(1f)) {
            Text(title ?: if (engine.phase == Phase.Empty) "Equity" else "New chat", style = MaterialTheme.typography.titleMedium, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Row(verticalAlignment = Alignment.CenterVertically) {
                StatusDot(engine.phase)
                Spacer(Modifier.width(6.dp))
                val detail = when (engine.phase) {
                    Phase.Empty -> "No model loaded"
                    Phase.Loading -> engine.loadPercent?.let { "Loading $it%" } ?: engine.status
                    Phase.Ready -> buildString {
                        append(engine.modelName ?: "Ready")
                        engine.lastSpeed?.let { append(" · last %.1f tok/s".format(it)) }
                    }
                    else -> engine.status
                }
                Text(detail, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, maxLines = 1,
                    overflow = TextOverflow.Ellipsis)
            }
        }
        val reload = Catalog.byId(engine.lastModelId)
        if (engine.phase == Phase.Empty && reload != null) {
            TextButton(onClick = { EngineController.load(context, reload) }) { Text("Load model") }
        }
        IconButton(onClick = onNewChat, enabled = canStartNew) { Icon(Icons.Rounded.Add, contentDescription = "New chat") }
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
            Phase.Generating -> Text("Equity is answering in another chat.", style = MaterialTheme.typography.bodyMedium,
                color = colors.onSurfaceVariant)
            Phase.Ready -> FlowRow(
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
}

@Composable
internal fun AssistantMessage(message: Message, generatedTokens: Int? = null) {
    val colors = MaterialTheme.colorScheme
    val clipboard = LocalClipboard.current
    val scope = rememberCoroutineScope()
    Column(Modifier.fillMaxWidth().padding(end = 12.dp)) {
        if (!message.text.hasVisibleContent()) {
            // Static: an animation here would compete with the prompt's evaluation for the big cores.
            val sampled = (generatedTokens ?: 0) > 0 || message.text.isNotEmpty()
            Text(when {
                !message.streaming -> "The model finished without producing visible text."
                sampled -> "Generating tokens; no visible text yet"
                else -> "Reading your message…"
            }, style = MaterialTheme.typography.bodyMedium, color = colors.onSurfaceVariant,
                modifier = Modifier.padding(vertical = 8.dp))
            if (message.streaming && generatedTokens != null && generatedTokens > 0) {
                Text("$generatedTokens tokens", style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant)
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
                val note = message.stats ?: if (message.failed) "Not completed" else null
                note?.let {
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
    steeringEnabled: Boolean,
    onView: (File) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val ready = engine.phase == Phase.Ready
    val generating = engine.phase == Phase.Generating
    var menu by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 6.dp, bottom = 6.dp)) {
        Surface(color = colors.surfaceContainerHigh, shape = RoundedCornerShape(28.dp)) {
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
                                        Icon(Icons.Rounded.Close, contentDescription = "Remove ${attachment.name}", modifier = Modifier.size(16.dp))
                                    }
                                }
                            }
                        }
                        if (importing) CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
                    }
                }
                Row(verticalAlignment = Alignment.Bottom, modifier = Modifier.padding(start = 4.dp, end = 6.dp, bottom = 6.dp)) {
                    Box {
                        IconButton(onClick = { menu = true }, enabled = ready || generating, modifier = Modifier.padding(bottom = 2.dp)) {
                            Icon(Icons.Outlined.AttachFile, contentDescription = "Attach")
                        }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }, containerColor = colors.surfaceContainerHighest) {
                            val imageHint = if (engine.vision) null else "Needs a model with image support"
                            DropdownMenuItem(
                                text = { MenuLabel("Photos", imageHint) },
                                leadingIcon = { Icon(Icons.Outlined.PhotoLibrary, contentDescription = null) },
                                enabled = engine.vision,
                                onClick = { menu = false; onPhotos() },
                            )
                            DropdownMenuItem(
                                text = { MenuLabel("Camera", imageHint) },
                                leadingIcon = { Icon(Icons.Outlined.PhotoCamera, contentDescription = null) },
                                enabled = engine.vision,
                                onClick = { menu = false; onCamera() },
                            )
                            DropdownMenuItem(
                                text = { MenuLabel("File", "Text, code or PDF") },
                                leadingIcon = { Icon(Icons.Outlined.Description, contentDescription = null) },
                                onClick = { menu = false; onFile() },
                            )
                        }
                    }
                    IconButton(onClick = onConfigure, modifier = Modifier.padding(bottom = 2.dp)) {
                        Icon(Icons.Outlined.Tune, contentDescription = "Chat configuration",
                            tint = if (steeringEnabled) colors.primary else colors.onSurfaceVariant)
                    }
                    TextField(
                        value = value,
                        onValueChange = onValueChange,
                        enabled = ready || generating,
                        placeholder = {
                            Text(when {
                                engine.phase == Phase.Empty -> "Load a model to start"
                                answeringElsewhere -> "Equity is answering in another chat"
                                else -> "Message Equity"
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
                        modifier = Modifier.weight(1f),
                    )
                    if (generating) {
                        FilledIconButton(
                            onClick = onStop,
                            colors = IconButtonDefaults.filledIconButtonColors(containerColor = colors.onSurface, contentColor = colors.surface),
                        ) { Icon(Icons.Rounded.Stop, contentDescription = "Stop") }
                    } else {
                        FilledIconButton(onClick = onSend, enabled = ready && !importing && (value.isNotBlank() || pending.isNotEmpty())) {
                            Icon(Icons.AutoMirrored.Rounded.Send, contentDescription = "Send")
                        }
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

@Composable
private fun MenuLabel(title: String, hint: String?) {
    Column {
        Text(title)
        hint?.let { Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
    }
}

@Composable
private fun RemoveBadge(modifier: Modifier, onClick: () -> Unit) {
    Surface(color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f), shape = CircleShape,
        modifier = modifier.padding(3.dp).size(22.dp), onClick = onClick) {
        Box(contentAlignment = Alignment.Center) {
            Icon(Icons.Rounded.Close, contentDescription = "Remove image", modifier = Modifier.size(14.dp))
        }
    }
}
