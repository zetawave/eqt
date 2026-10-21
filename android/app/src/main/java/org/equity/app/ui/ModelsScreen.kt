package org.equity.app.ui

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.FolderOpen
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.SdStorage
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.ExpandLess
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.work.WorkInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.equity.app.Artifact
import org.equity.app.Catalog
import org.equity.app.CatalogModel
import org.equity.app.ChatStore
import org.equity.app.Downloads
import org.equity.app.EngineController
import org.equity.app.EngineState
import org.equity.app.ModelFiles
import org.equity.app.Phase
import org.equity.app.SteeringFiles
import org.equity.app.SteeringSummary
import org.equity.app.formatBytes

@Composable
fun ModelsScreen(engine: EngineState, onOpenChat: () -> Unit) {
    val context = LocalContext.current
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) EngineController.load(context, uri)
    }
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            Column(Modifier.statusBarsPadding().padding(top = 20.dp, start = 4.dp, bottom = 2.dp)) {
                Text("Models", style = MaterialTheme.typography.headlineMedium)
                Text("Download once, then run offline. Files are checked against pinned fingerprints.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        item { DeviceCard() }
        items(Catalog.models, key = { it.id }) { model -> ModelCard(model, engine, onOpenChat) }
        // A local file is loaded (catalog models have their own card); a steering operation keeps it shown.
        if (engine.modelId == null && engine.modelName != null &&
            (engine.phase == Phase.Ready || engine.phase == Phase.Generating || engine.steeringOperation)) {
            item { LocalModelCard(engine) }
        }
        item {
            val enabled = engine.phase == Phase.Empty || engine.phase == Phase.Ready
            Surface(color = MaterialTheme.colorScheme.surfaceContainerLow, shape = RoundedCornerShape(20.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
                Row(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.FolderOpen, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text("Open a local GGUF", style = MaterialTheme.typography.titleSmall)
                        Text("Qwen3, Qwen3.5/3.6 or Qwen3.8 files on this phone",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }, enabled = enabled) { Text("Open") }
                }
            }
        }
    }
}

/** Free memory in bytes, refreshed every few seconds while shown. */
@Composable
private fun rememberAvailableMemory(): Long {
    val context = LocalContext.current
    val available by produceState(0L) {
        val manager = context.getSystemService(ActivityManager::class.java)
        while (true) {
            value = ActivityManager.MemoryInfo().also { manager.getMemoryInfo(it) }.availMem
            delay(3000)
        }
    }
    return available
}

/**
 * Free memory against what [model] needs at full speed, with the way to free more when it falls short. With
 * [shortfallOnly] nothing is shown while there is enough.
 */
@Composable
private fun MemoryNeed(model: CatalogModel, shortfallOnly: Boolean = false) {
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    val available = rememberAvailableMemory()
    // Recomputed on every refresh: it depends on the context size and processor chosen in Settings.
    val needed = EngineController.fullSpeedMemoryMiB(model).toLong() shl 20
    var explain by remember { mutableStateOf(false) }
    if (available <= 0) return
    val enough = available >= needed
    if (enough && shortfallOnly) return
    Spacer(Modifier.height(8.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            if (enough) "${formatBytes(available)} free · full speed needs ${formatBytes(needed)}"
            else "${formatBytes(available)} free of the ${formatBytes(needed)} full speed needs; it will load with slower prompts or not at all",
            style = MaterialTheme.typography.bodySmall,
            color = if (enough) colors.onSurfaceVariant else colors.error,
            modifier = Modifier.weight(1f),
        )
        if (!enough) TextButton(onClick = { explain = true }) { Text("Free memory") }
    }
    if (explain) {
        val developer = Settings.Global.getInt(context.contentResolver, Settings.Global.DEVELOPMENT_SETTINGS_ENABLED, 0) != 0
        AlertDialog(
            onDismissRequest = { explain = false },
            title = { Text("Free memory for ${model.name}") },
            text = {
                Text(
                    if (developer) "In Developer options, set Background process limit to No background processes. " +
                        "Android then stops keeping closed apps in memory. Come back here when it is set. The limit " +
                        "lasts until the phone restarts; other apps start from scratch when you open them again."
                    else "Developer options are turned off. Open About phone, then Software information, and tap " +
                        "Build number seven times. Then come back and tap Free memory again."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    explain = false
                    openSettings(context, if (developer) Settings.ACTION_APPLICATION_DEVELOPMENT_SETTINGS
                        else Settings.ACTION_DEVICE_INFO_SETTINGS)
                }) { Text(if (developer) "Open Developer options" else "Open About phone") }
            },
            dismissButton = { TextButton(onClick = { explain = false }) { Text("Cancel") } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

private fun openSettings(context: Context, action: String) {
    // Some builds lack a specific page; the main Settings screen still gets the user there.
    runCatching { context.startActivity(Intent(action)) }
        .recoverCatching { context.startActivity(Intent(Settings.ACTION_SETTINGS)) }
}

@Composable
private fun DeviceCard() {
    val context = LocalContext.current
    val snapshot by produceState(Triple(0L, 0L, 0L)) {
        val manager = context.getSystemService(ActivityManager::class.java)
        while (true) {
            val memory = ActivityManager.MemoryInfo().also { manager.getMemoryInfo(it) }
            value = Triple(memory.availMem, memory.totalMem, ModelFiles.freeBytes(context))
            delay(3000)
        }
    }
    val (available, total, storage) = snapshot
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, shape = RoundedCornerShape(20.dp)) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Metric(Icons.Outlined.Memory, if (total > 0) "${formatBytes(available)} of ${formatBytes(total)} free" else "…",
                "Memory", Modifier.weight(1f))
            Metric(Icons.Outlined.SdStorage, if (storage > 0) "${formatBytes(storage)} free" else "…", "Storage", Modifier.weight(1f))
        }
    }
}

@Composable
private fun Metric(icon: ImageVector, value: String, label: String, modifier: Modifier) {
    Row(modifier, verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, contentDescription = label, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(8.dp))
        Text(value, style = MaterialTheme.typography.bodySmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ModelCard(model: CatalogModel, engine: EngineState, onOpenChat: () -> Unit) {
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    val work by remember(model.id) { Downloads.observe(context, model.artifact) }.collectAsStateWithLifecycle(initialValue = null)
    var refresh by remember { mutableIntStateOf(0) }
    val state = work?.state
    val installed = remember(state, refresh) { ModelFiles.installed(context, model.artifact) }
    val partial = remember(state, refresh, work?.progress) { ModelFiles.partialBytes(context, model.artifact) }
    val transferring = state == WorkInfo.State.RUNNING || state == WorkInfo.State.ENQUEUED || state == WorkInfo.State.BLOCKED
    val loaded = engine.modelId == model.id && (engine.phase == Phase.Ready || engine.phase == Phase.Generating)
    val loading = engine.modelId == model.id && engine.phase == Phase.Loading
    val engineIdle = engine.phase == Phase.Empty || engine.phase == Phase.Ready
    var confirmDelete by remember { mutableStateOf(false) }
    var details by rememberSaveable(model.id) { mutableStateOf(false) }
    var options by remember { mutableStateOf(false) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { Downloads.start(context, model.artifact) }
    val startDownload = {
        val needsPermission = Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        if (needsPermission) permission.launch(Manifest.permission.POST_NOTIFICATIONS) else Downloads.start(context, model.artifact)
    }
    LaunchedEffect(state) { if (state?.isFinished == true) refresh++ }

    Card(
        colors = CardDefaults.cardColors(containerColor = colors.surfaceContainer),
        border = if (loaded) BorderStroke(1.dp, colors.primary.copy(alpha = 0.6f)) else null,
        shape = RoundedCornerShape(20.dp),
    ) {
        Column(Modifier.padding(start = 16.dp, end = 8.dp, top = 12.dp, bottom = 6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(36.dp).background(BrandGradient, CircleShape), contentAlignment = Alignment.Center) {
                    Text(model.badge, color = colors.surface, fontWeight = FontWeight.Bold, fontSize = 10.sp, fontFamily = Inter)
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(model.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(
                        listOfNotNull(formatBytes(model.sizeBytes), if (model.streamed) "streamed" else "in memory",
                            "images".takeIf { model.projector != null }).joinToString(" · "),
                        style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, maxLines = 1,
                    )
                }
                if (loaded) Pill("Active", colors.primaryContainer, colors.onPrimaryContainer)
                else if (installed) Icon(Icons.Outlined.CheckCircle, contentDescription = "Downloaded", tint = colors.primary,
                    modifier = Modifier.size(20.dp))
                if (model.projector != null || installed || partial > 0) IconButton(onClick = { options = true }) {
                    Icon(Icons.Outlined.MoreVert, contentDescription = "Options for ${model.name}", tint = colors.onSurfaceVariant)
                } else Spacer(Modifier.width(12.dp))
            }
            Column(Modifier.padding(end = 8.dp, top = 8.dp)) {
                when {
                    loading -> {
                        val percent = engine.loadPercent
                        if (percent != null) ThinProgress(percent / 100f) else ThinProgress(null)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(engine.status, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, modifier = Modifier.weight(1f))
                            TextButton(onClick = { EngineController.stop() }) { Text("Cancel") }
                        }
                    }
                    transferring -> {
                        DownloadProgress(model.artifact, work)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick = { Downloads.pause(context, model.artifact) }) {
                                Icon(Icons.Rounded.Pause, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("Pause")
                            }
                            TextButton(onClick = { confirmDelete = true }) { Text("Cancel download") }
                        }
                    }
                    installed && loaded -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = onOpenChat) { Text("Open chat") }
                        OutlinedButton(onClick = { EngineController.unload() }, enabled = engine.phase == Phase.Ready) { Text("Unload") }
                    }
                    installed -> {
                        Button(onClick = { EngineController.load(context, model) }, enabled = engineIdle) {
                            Text(if (engine.phase == Phase.Ready) "Switch to this model" else "Load")
                        }
                        if (model.streamed) MemoryNeed(model, shortfallOnly = true)
                    }
                    else -> {
                        if (state == WorkInfo.State.FAILED) {
                            Text(work?.outputData?.getString(Downloads.KEY_ERROR) ?: "Download failed",
                                style = MaterialTheme.typography.bodySmall, color = colors.error)
                            Spacer(Modifier.height(6.dp))
                        }
                        if (partial > 0) {
                            ThinProgress(partial.toFloat() / model.sizeBytes)
                            Spacer(Modifier.height(4.dp))
                            Text("Paused at ${formatBytes(partial)} of ${formatBytes(model.sizeBytes)}",
                                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                            Spacer(Modifier.height(6.dp))
                        }
                        Button(onClick = startDownload) {
                            Icon(Icons.Rounded.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text(if (partial > 0) "Resume" else "Download · ${formatBytes(model.sizeBytes)}")
                        }
                    }
                }
            }
            // Details that are not needed to load or download: collapsed until asked for.
            Row(
                Modifier.fillMaxWidth().clickable { details = !details }.padding(top = 4.dp, bottom = 4.dp, end = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(model.tagline, style = MaterialTheme.typography.labelMedium, color = colors.onSurfaceVariant,
                    maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
                Icon(if (details) Icons.Rounded.ExpandLess else Icons.Rounded.ExpandMore,
                    contentDescription = if (details) "Hide details" else "Show details", tint = colors.onSurfaceVariant)
            }
            AnimatedVisibility(details) {
                Column(Modifier.padding(end = 8.dp, bottom = 8.dp)) {
                    Text(model.description, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    Spacer(Modifier.height(8.dp))
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Pill(if (model.streamed) "Expert streaming" else "In memory", colors.surfaceContainerHighest, colors.onSurface)
                        if (model.projector != null) Pill("Images", colors.surfaceContainerHighest, colors.onSurface)
                        Pill(model.license, colors.surfaceContainerHighest, colors.onSurfaceVariant)
                    }
                    if (model.streamed && installed) MemoryNeed(model)
                    if (model.streamed && !installed && partial == 0L) {
                        Spacer(Modifier.height(6.dp))
                        Text("Downloads over Wi-Fi only and continues in the background.",
                            style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    }
                }
            }
        }
    }

    if (options) ModelOptionsSheet(
        title = model.name,
        subtitle = "${model.license} · ${formatBytes(model.sizeBytes)}",
        onDismiss = { options = false },
    ) {
        model.projector?.let { ProjectorSection(model, it, engine) }
        if (installed) SteeringSection(model.sha256, engine, loaded)
        if ((installed || partial > 0) && !transferring) {
            Spacer(Modifier.height(18.dp))
            HorizontalDivider(color = colors.outlineVariant)
            Spacer(Modifier.height(6.dp))
            TextButton(onClick = { options = false; confirmDelete = true }, enabled = !loading) {
                Icon(Icons.Outlined.Delete, contentDescription = null, modifier = Modifier.size(18.dp), tint = colors.error)
                Spacer(Modifier.width(6.dp))
                Text(if (installed) "Delete model" else "Discard download", color = colors.error)
            }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(if (installed) "Delete ${model.name}?" else "Discard the download?") },
            text = { Text("This frees up to ${formatBytes(model.sizeBytes)}. You can download it again later.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    if (engine.modelId == model.id) EngineController.unload()
                    if (installed) ChatStore.deleteSessionsOf(model.sha256)
                    Downloads.delete(context, model.artifact) { refresh++ }
                }) { Text("Delete", color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Keep") } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

/** A model's settings that are not needed to load it (image support, steering, deletion), in a sheet of their own. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelOptionsSheet(title: String, subtitle: String, onDismiss: () -> Unit, content: @Composable () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss, containerColor = MaterialTheme.colorScheme.surfaceContainerLow) {
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(start = 20.dp, end = 20.dp, bottom = 28.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            content()
        }
    }
}

/** A GGUF opened from the phone's files: shown while loaded, for its steering vector. */
@Composable
private fun LocalModelCard(engine: EngineState) {
    val colors = MaterialTheme.colorScheme
    var options by remember { mutableStateOf(false) }
    Card(
        colors = CardDefaults.cardColors(containerColor = colors.surfaceContainer),
        border = BorderStroke(1.dp, colors.primary.copy(alpha = 0.6f)),
        shape = RoundedCornerShape(20.dp),
    ) {
        Row(Modifier.padding(start = 16.dp, end = 8.dp, top = 10.dp, bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(engine.modelName ?: "Local model", style = MaterialTheme.typography.titleSmall)
                Text("Opened from this phone's files", style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
            }
            Pill("Active", colors.primaryContainer, colors.onPrimaryContainer)
            IconButton(onClick = { options = true }) {
                Icon(Icons.Outlined.MoreVert, contentDescription = "Options for the local model", tint = colors.onSurfaceVariant)
            }
        }
    }
    if (options) ModelOptionsSheet(engine.modelName ?: "Local model", "Opened from this phone's files", { options = false }) {
        SteeringSection(engine.modelSha256, engine, loaded = true)
    }
}

/**
 * The directional steering vector of one exact model file (by fingerprint): imported here and offered in that
 * model's chats while it is loaded.
 */
@Composable
private fun SteeringSection(modelHash: String?, engine: EngineState, loaded: Boolean) {
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    val summary by produceState<SteeringSummary?>(null, modelHash, engine.steeringRevision) {
        value = modelHash?.let { hash -> withContext(Dispatchers.IO) { SteeringFiles.summary(context, hash) } }
    }
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) EngineController.importSteering(context, uris, modelHash)
    }
    var confirmRemove by remember { mutableStateOf(false) }
    val current = summary
    Spacer(Modifier.height(18.dp))
    HorizontalDivider(color = colors.outlineVariant)
    Spacer(Modifier.height(14.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Outlined.Tune, contentDescription = null, tint = colors.primary, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Text("Directional steering", style = MaterialTheme.typography.titleSmall)
            Text(
                when {
                    modelHash == null -> "Verify this file's fingerprint to import a vector for it."
                    current == null -> "Import a vector made for this exact file to steer its answers. Chats offer it while the model is loaded."
                    loaded && engine.steeringProfile == null && engine.steeringNote != null -> engine.steeringNote
                    else -> "Vector imported · layer ${current.layer}" +
                        (current.direction.takeIf { it.isNotEmpty() }?.let { " · ${it.replace("_minus_", " − ").replace('_', ' ')}" } ?: "") +
                        if (loaded) ". Turn it on per chat with the configuration button." else ". Available in chats once this model is loaded."
                },
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
            )
        }
    }
    if (engine.steeringMessageModel == modelHash) engine.steeringImportMessage?.let {
        Spacer(Modifier.height(6.dp))
        Text(it, style = MaterialTheme.typography.bodySmall, color = colors.primary)
    }
    Spacer(Modifier.height(10.dp))
    when {
        engine.steeringOperation && engine.steeringMessageModel == modelHash -> Row(verticalAlignment = Alignment.CenterVertically) {
            Text(engine.status, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, modifier = Modifier.weight(1f))
            TextButton(onClick = { EngineController.stop() }) { Text("Cancel") }
        }
        modelHash == null -> Column {
            OutlinedButton(onClick = { EngineController.verifySteeringModel(context) }, enabled = engine.phase == Phase.Ready) {
                Text("Verify local model")
            }
            Text("Reads and reloads the model once to check its exact fingerprint.",
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
        }
        else -> Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = { importer.launch(arrayOf("*/*")) }, enabled = !engine.busy) {
                    Text(if (current == null) "Import vector + JSON" else "Replace vector")
                }
                if (current != null) {
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = { confirmRemove = true }, enabled = !engine.busy) { Text("Remove") }
                }
            }
            Text("Select the vector .gguf and its matching .gguf.json together.",
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
        }
    }
    if (confirmRemove && modelHash != null) {
        AlertDialog(
            onDismissRequest = { confirmRemove = false },
            title = { Text("Remove the steering vector?") },
            text = { Text("Chats of this model stop offering steering. The vector can be imported again later.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmRemove = false
                    EngineController.removeSteering(context, modelHash)
                }) { Text("Remove", color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmRemove = false }) { Text("Keep") } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

/** Image input: the model's vision projector, downloaded and removed on its own. */
@Composable
private fun ProjectorSection(model: CatalogModel, projector: Artifact, engine: EngineState) {
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    val work by remember(projector.id) { Downloads.observe(context, projector) }.collectAsStateWithLifecycle(initialValue = null)
    var refresh by remember { mutableIntStateOf(0) }
    val state = work?.state
    val installed = remember(state, refresh) { ModelFiles.installed(context, projector) }
    val partial = remember(state, refresh, work?.progress) { ModelFiles.partialBytes(context, projector) }
    val transferring = state == WorkInfo.State.RUNNING || state == WorkInfo.State.ENQUEUED || state == WorkInfo.State.BLOCKED
    val loaded = engine.modelId == model.id && (engine.phase == Phase.Ready || engine.phase == Phase.Generating)
    var confirmDelete by remember { mutableStateOf(false) }
    LaunchedEffect(state) { if (state?.isFinished == true) refresh++ }
    Spacer(Modifier.height(18.dp))
    HorizontalDivider(color = colors.outlineVariant)
    Spacer(Modifier.height(14.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(Icons.Outlined.Image, contentDescription = null, tint = colors.primary, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            Text("Image input", style = MaterialTheme.typography.titleSmall)
            Text(
                when {
                    installed && loaded && !engine.vision -> "Downloaded · load the model again to use it"
                    installed -> "Ready · photos and scanned PDFs can be attached"
                    else -> "Lets this model read photos, screenshots and scanned documents. While it reads an image " +
                        "it needs about ${formatBytes(projector.sizeBytes + (512L shl 20))} of extra memory."
                },
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant,
            )
        }
    }
    Spacer(Modifier.height(10.dp))
    when {
        transferring -> {
            DownloadProgress(projector, work)
            Spacer(Modifier.height(4.dp))
            TextButton(onClick = { Downloads.pause(context, projector) }) { Text("Pause") }
        }
        installed -> TextButton(onClick = { confirmDelete = true }, enabled = engine.phase != Phase.Generating) {
            Icon(Icons.Outlined.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text("Remove image support")
        }
        else -> Column {
            if (state == WorkInfo.State.FAILED) {
                Text(work?.outputData?.getString(Downloads.KEY_ERROR) ?: "Download failed",
                    style = MaterialTheme.typography.bodySmall, color = colors.error)
                Spacer(Modifier.height(6.dp))
            }
            OutlinedButton(onClick = { Downloads.start(context, projector) }) {
                Icon(Icons.Rounded.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text(if (partial > 0) "Resume · ${formatBytes(partial)} of ${formatBytes(projector.sizeBytes)}"
                    else "Add image support · ${formatBytes(projector.sizeBytes)}")
            }
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Remove image support?") },
            text = { Text("This frees ${formatBytes(projector.sizeBytes)}. Models that share it lose image input until it is downloaded again.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    // The engine reads the projector on every image request; release a model loaded with it first.
                    if (Catalog.byId(engine.modelId)?.projector?.id == projector.id && engine.vision) EngineController.unload()
                    Downloads.delete(context, projector) { refresh++ }
                }) { Text("Remove", color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Keep") } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

@Composable
private fun DownloadProgress(model: Artifact, work: WorkInfo?) {
    val colors = MaterialTheme.colorScheme
    val progress = work?.progress
    val done = progress?.getLong(Downloads.KEY_DONE, 0L) ?: 0L
    val total = progress?.getLong(Downloads.KEY_TOTAL, model.sizeBytes)?.takeIf { it > 0 } ?: model.sizeBytes
    val speed = progress?.getLong(Downloads.KEY_SPEED, 0L) ?: 0L
    val verifying = progress?.getString(Downloads.KEY_PHASE) == Downloads.PHASE_VERIFYING
    val waiting = work?.state != WorkInfo.State.RUNNING
    if (waiting) ThinProgress(null) else ThinProgress(done.toFloat() / total)
    Spacer(Modifier.height(8.dp))
    val text = when {
        waiting -> if (model.unmeteredOnly) "Waiting for Wi-Fi…" else "Waiting for a connection…"
        verifying -> "Checking the partial download · ${formatBytes(done)}"
        else -> buildString {
            append("${formatBytes(done)} of ${formatBytes(total)}")
            if (speed > 0) {
                append(" · ${formatBytes(speed)}/s")
                val minutes = (total - done) / speed / 60
                append(if (minutes >= 1) " · $minutes min left" else " · under a minute left")
            }
        }
    }
    Text(text, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
}

@Composable
fun ThinProgress(progress: Float?) {
    val modifier = Modifier.fillMaxWidth().height(6.dp)
    val track = MaterialTheme.colorScheme.surfaceContainerHighest
    if (progress == null) LinearProgressIndicator(modifier, trackColor = track)
    else LinearProgressIndicator(progress = { progress.coerceIn(0f, 1f) }, modifier = modifier, trackColor = track, gapSize = 0.dp, drawStopIndicator = {})
}

@Composable
fun Pill(text: String, container: androidx.compose.ui.graphics.Color, content: androidx.compose.ui.graphics.Color) {
    Surface(color = container, shape = RoundedCornerShape(50)) {
        Text(text, style = MaterialTheme.typography.labelSmall, color = content,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp))
    }
}
