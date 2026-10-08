package org.equity.app.ui

import android.Manifest
import android.app.ActivityManager
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.FolderOpen
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material.icons.outlined.SdStorage
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.work.WorkInfo
import kotlinx.coroutines.delay
import org.equity.app.Catalog
import org.equity.app.CatalogModel
import org.equity.app.Downloads
import org.equity.app.EngineController
import org.equity.app.EngineState
import org.equity.app.ModelFiles
import org.equity.app.Phase
import org.equity.app.formatBytes

@Composable
fun ModelsScreen(engine: EngineState, onOpenChat: () -> Unit) {
    val context = LocalContext.current
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) EngineController.load(context, uri)
    }
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Column(Modifier.statusBarsPadding().padding(top = 20.dp)) {
                Text("Models", style = MaterialTheme.typography.headlineMedium)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Download once, then run fully offline. Files stay in Equity's private storage and are checked against a pinned fingerprint.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        item { DeviceCard() }
        items(Catalog.models, key = { it.id }) { model -> ModelCard(model, engine, onOpenChat) }
        item {
            val enabled = engine.phase == Phase.Empty || engine.phase == Phase.Ready
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                shape = RoundedCornerShape(24.dp),
            ) {
                Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.FolderOpen, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(Modifier.width(16.dp))
                    Column(Modifier.weight(1f)) {
                        Text("Open a local GGUF", style = MaterialTheme.typography.titleSmall)
                        Text("Use a Qwen3 or Qwen3.5/3.6 file already on this phone. Large MoE files stream their experts automatically.",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Spacer(Modifier.width(12.dp))
                    OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }, enabled = enabled) { Text("Open") }
                }
            }
        }
    }
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
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Metric(Icons.Outlined.Memory, "Free memory", if (total > 0) "${formatBytes(available)} of ${formatBytes(total)}" else "…", Modifier.weight(1f))
        Metric(Icons.Outlined.SdStorage, "Free storage", if (storage > 0) formatBytes(storage) else "…", Modifier.weight(1f))
    }
}

@Composable
private fun Metric(icon: ImageVector, label: String, value: String, modifier: Modifier) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, shape = RoundedCornerShape(20.dp), modifier = modifier) {
        Column(Modifier.padding(16.dp)) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
            Spacer(Modifier.height(10.dp))
            Text(value, style = MaterialTheme.typography.titleSmall)
            Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun ModelCard(model: CatalogModel, engine: EngineState, onOpenChat: () -> Unit) {
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    val work by remember(model.id) { Downloads.observe(context, model) }.collectAsStateWithLifecycle(initialValue = null)
    var refresh by remember { mutableIntStateOf(0) }
    val state = work?.state
    val installed = remember(state, refresh) { ModelFiles.installed(context, model) }
    val partial = remember(state, refresh, work?.progress) { ModelFiles.partialBytes(context, model) }
    val transferring = state == WorkInfo.State.RUNNING || state == WorkInfo.State.ENQUEUED || state == WorkInfo.State.BLOCKED
    val loaded = engine.modelId == model.id && (engine.phase == Phase.Ready || engine.phase == Phase.Generating)
    val loading = engine.modelId == model.id && engine.phase == Phase.Loading
    val engineIdle = engine.phase == Phase.Empty || engine.phase == Phase.Ready
    var confirmDelete by remember { mutableStateOf(false) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { Downloads.start(context, model) }
    val startDownload = {
        val needsPermission = Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        if (needsPermission) permission.launch(Manifest.permission.POST_NOTIFICATIONS) else Downloads.start(context, model)
    }
    LaunchedEffect(state) { if (state?.isFinished == true) refresh++ }

    Card(
        colors = CardDefaults.cardColors(containerColor = colors.surfaceContainer),
        border = if (loaded) BorderStroke(1.dp, colors.primary.copy(alpha = 0.6f)) else null,
        shape = RoundedCornerShape(24.dp),
    ) {
        Column(Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(44.dp).background(BrandGradient, CircleShape), contentAlignment = Alignment.Center) {
                    Text(if (model.streamed) "35B" else "0.6B", color = colors.surface, fontWeight = FontWeight.Bold,
                        fontSize = 12.sp, fontFamily = Inter)
                }
                Spacer(Modifier.width(14.dp))
                Column(Modifier.weight(1f)) {
                    Text(model.name, style = MaterialTheme.typography.titleMedium)
                    Text(model.tagline, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                }
                if (loaded) Pill("Active", colors.primaryContainer, colors.onPrimaryContainer)
                else if (installed) Icon(Icons.Outlined.CheckCircle, contentDescription = "Downloaded", tint = colors.primary)
            }
            Spacer(Modifier.height(14.dp))
            Text(model.description, style = MaterialTheme.typography.bodyMedium, color = colors.onSurfaceVariant)
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Pill(formatBytes(model.sizeBytes), colors.surfaceContainerHighest, colors.onSurface)
                Pill(if (model.streamed) "Expert streaming" else "In memory", colors.surfaceContainerHighest, colors.onSurface)
                Pill(model.license, colors.surfaceContainerHighest, colors.onSurfaceVariant)
            }
            Spacer(Modifier.height(18.dp))
            when {
                loading -> {
                    val percent = engine.loadPercent
                    if (percent != null) ThinProgress(percent / 100f) else ThinProgress(null)
                    Spacer(Modifier.height(8.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(engine.status, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, modifier = Modifier.weight(1f))
                        TextButton(onClick = { EngineController.stop() }) { Text("Cancel") }
                    }
                }
                transferring -> {
                    DownloadProgress(model, work)
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = { Downloads.pause(context, model) }) {
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
                installed -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(onClick = { EngineController.load(context, model) }, enabled = engineIdle) {
                        Text(if (engine.phase == Phase.Ready) "Switch to this model" else "Load")
                    }
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = { confirmDelete = true }) {
                        Icon(Icons.Outlined.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("Delete")
                    }
                }
                else -> Column {
                    if (state == WorkInfo.State.FAILED) {
                        Text(work?.outputData?.getString(Downloads.KEY_ERROR) ?: "Download failed",
                            style = MaterialTheme.typography.bodySmall, color = colors.error)
                        Spacer(Modifier.height(8.dp))
                    }
                    if (partial > 0) {
                        ThinProgress(partial.toFloat() / model.sizeBytes)
                        Spacer(Modifier.height(6.dp))
                        Text("Paused at ${formatBytes(partial)} of ${formatBytes(model.sizeBytes)}",
                            style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                        Spacer(Modifier.height(8.dp))
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Button(onClick = startDownload) {
                            Icon(Icons.Rounded.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text(if (partial > 0) "Resume" else "Download · ${formatBytes(model.sizeBytes)}")
                        }
                        if (partial > 0) {
                            Spacer(Modifier.weight(1f))
                            TextButton(onClick = { confirmDelete = true }) { Text("Discard") }
                        }
                    }
                    if (model.streamed && partial == 0L) {
                        Spacer(Modifier.height(8.dp))
                        Text("Downloads over Wi-Fi only and continues in the background.",
                            style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    }
                }
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
                    Downloads.delete(context, model) { refresh++ }
                }) { Text("Delete", color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Keep") } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

@Composable
private fun DownloadProgress(model: CatalogModel, work: WorkInfo?) {
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
        waiting -> if (model.streamed) "Waiting for Wi-Fi…" else "Waiting for a connection…"
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
