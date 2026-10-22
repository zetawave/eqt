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
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Download
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
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import org.equity.app.R
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
    val strings = LocalContext.current.resources
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
                Text(strings.getString(R.string.models), style = MaterialTheme.typography.headlineMedium)
                Text(
                    strings.getString(R.string.download_once_then_run_offline_files_are_checked_against_pinned_fingerprint),
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
            Surface(color = MaterialTheme.colorScheme.surfaceContainerLow.copy(alpha = 0.78f), shape = RoundedCornerShape(20.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
                Row(Modifier.padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.FolderOpen, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(strings.getString(R.string.open_a_local_gguf), style = MaterialTheme.typography.titleSmall)
                        Text(strings.getString(R.string.qwen3_qwen3_5_3_6_or_qwen3_8_files_on_this_phone),
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    PremiumOutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }, enabled = enabled, compact = true) {
                        Text(strings.getString(R.string.open), style = MaterialTheme.typography.labelLarge)
                    }
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
    val strings = LocalContext.current.resources
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
            if (enough) strings.getString(R.string.value_1_s_free_full_speed_needs_2_s, formatBytes(available), formatBytes(needed))
            else strings.getString(R.string.value_1_s_free_of_the_2_s_full_speed_needs_it_will_load_with_slower_prompts_or_no, formatBytes(available), formatBytes(needed)),
            style = MaterialTheme.typography.bodySmall,
            color = if (enough) colors.onSurfaceVariant else colors.error,
            modifier = Modifier.weight(1f),
        )
        if (!enough) TextButton(onClick = { explain = true }) { Text(strings.getString(R.string.free_memory)) }
    }
    if (explain) {
        val developer = Settings.Global.getInt(context.contentResolver, Settings.Global.DEVELOPMENT_SETTINGS_ENABLED, 0) != 0
        AlertDialog(
            onDismissRequest = { explain = false },
            title = { Text(strings.getString(R.string.free_memory_for_1_s, model.name)) },
            text = {
                Text(
                    if (developer) strings.getString(R.string.in_developer_options_set_background_process_limit_to_no_background_processe)
                    else strings.getString(R.string.developer_options_are_turned_off_open_about_phone_then_software_information)
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    explain = false
                    openSettings(context, if (developer) Settings.ACTION_APPLICATION_DEVELOPMENT_SETTINGS
                        else Settings.ACTION_DEVICE_INFO_SETTINGS)
                }) { Text(if (developer) strings.getString(R.string.open_developer_options) else strings.getString(R.string.open_about_phone)) }
            },
            dismissButton = { TextButton(onClick = { explain = false }) { Text(strings.getString(R.string.cancel)) } },
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
    val strings = LocalContext.current.resources
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
    Surface(
        color = MaterialTheme.colorScheme.surfaceContainerLow.copy(alpha = 0.78f),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
    ) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Metric(Icons.Outlined.Memory, if (total > 0) strings.getString(R.string.value_1_s_of_2_s_free, formatBytes(available), formatBytes(total)) else "…",
                strings.getString(R.string.memory), Modifier.weight(1f))
            Metric(Icons.Outlined.SdStorage, if (storage > 0) strings.getString(R.string.value_1_s_free, formatBytes(storage)) else "…", strings.getString(R.string.storage), Modifier.weight(1f))
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
    val strings = LocalContext.current.resources
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
        colors = CardDefaults.cardColors(containerColor = colors.surfaceContainerLow.copy(alpha = 0.8f)),
        border = if (loaded) BorderStroke(1.dp, BrandGradient) else BorderStroke(1.dp, colors.outlineVariant),
        shape = RoundedCornerShape(24.dp),
    ) {
        Column(Modifier.padding(start = 16.dp, end = 8.dp, top = 12.dp, bottom = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(36.dp).background(BrandGradient, CircleShape), contentAlignment = Alignment.Center) {
                    Text(model.badge, color = colors.surface, fontWeight = FontWeight.Bold, fontSize = 10.sp, fontFamily = Inter)
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(model.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(
                        listOfNotNull(formatBytes(model.sizeBytes), if (model.streamed) strings.getString(R.string.streamed) else strings.getString(R.string.in_memory),
                            strings.getString(R.string.images).takeIf { model.projector != null }).joinToString(" · "),
                        style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, maxLines = 1,
                    )
                }
                if (loaded) Pill(strings.getString(R.string.active), colors.primaryContainer, colors.onPrimaryContainer)
                else if (installed) Icon(Icons.Outlined.CheckCircle, contentDescription = strings.getString(R.string.downloaded), tint = colors.primary,
                    modifier = Modifier.size(20.dp))
                if (model.projector != null || installed || partial > 0) IconButton(onClick = { options = true }) {
                    Icon(Icons.Outlined.MoreVert, contentDescription = strings.getString(R.string.options_for_1_s, model.name), tint = colors.onSurfaceVariant)
                } else Spacer(Modifier.width(12.dp))
            }
            Column(Modifier.fillMaxWidth().padding(end = 8.dp, top = 8.dp)) {
                when {
                    loading -> {
                        val percent = engine.loadPercent
                        if (percent != null) ThinProgress(percent / 100f) else ThinProgress(null)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(engine.status, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant, modifier = Modifier.weight(1f))
                            TextButton(onClick = { EngineController.stop() }) { Text(strings.getString(R.string.cancel)) }
                        }
                    }
                    transferring -> {
                        DownloadProgress(model.artifact, work)
                        FlowRow(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
                            TextButton(onClick = { Downloads.pause(context, model.artifact) }) {
                                Icon(Icons.Rounded.Pause, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text(strings.getString(R.string.pause))
                            }
                            TextButton(onClick = { confirmDelete = true }) { Text(strings.getString(R.string.cancel_download)) }
                        }
                    }
                    installed && loaded -> FlowRow(Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
                        PremiumOutlinedButton(onClick = { EngineController.unload() }, enabled = engine.phase == Phase.Ready, compact = true) {
                            Text(strings.getString(R.string.unload), style = MaterialTheme.typography.labelLarge)
                        }
                        GradientButton(onClick = onOpenChat, icon = Icons.AutoMirrored.Filled.ArrowForward, compact = true) {
                            Text(strings.getString(R.string.open_chat), style = MaterialTheme.typography.labelLarge)
                        }
                    }
                    installed -> {
                        GradientButton(onClick = { EngineController.load(context, model) }, enabled = engineIdle,
                            modifier = Modifier.align(Alignment.End), compact = true,
                            icon = Icons.AutoMirrored.Filled.ArrowForward) {
                            Text(if (engine.phase == Phase.Ready) strings.getString(R.string.switch_to_this_model) else strings.getString(R.string.load),
                                style = MaterialTheme.typography.labelLarge)
                        }
                        if (model.streamed) MemoryNeed(model, shortfallOnly = true)
                    }
                    else -> {
                        if (state == WorkInfo.State.FAILED) {
                            Text(work?.outputData?.getString(Downloads.KEY_ERROR) ?: strings.getString(R.string.download_failed),
                                style = MaterialTheme.typography.bodySmall, color = colors.error)
                            Spacer(Modifier.height(6.dp))
                        }
                        if (partial > 0) {
                            ThinProgress(partial.toFloat() / model.sizeBytes)
                            Spacer(Modifier.height(4.dp))
                            Text(strings.getString(R.string.paused_at_1_s_of_2_s, formatBytes(partial), formatBytes(model.sizeBytes)),
                                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                            Spacer(Modifier.height(6.dp))
                        }
                        GradientButton(onClick = startDownload, icon = Icons.Filled.Download,
                            modifier = Modifier.align(Alignment.End), compact = true) {
                            Text(if (partial > 0) strings.getString(R.string.resume) else strings.getString(R.string.download_1_s, formatBytes(model.sizeBytes)),
                                style = MaterialTheme.typography.labelLarge)
                        }
                    }
                }
            }
            Row(
                Modifier.fillMaxWidth().clickable { details = !details }.padding(top = 8.dp, bottom = 4.dp, end = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(strings.getString(model.taglineRes), style = MaterialTheme.typography.labelMedium, color = colors.onSurfaceVariant,
                    modifier = Modifier.weight(1f))
                Icon(if (details) Icons.Rounded.ExpandLess else Icons.Rounded.ExpandMore,
                    contentDescription = if (details) strings.getString(R.string.hide_details) else strings.getString(R.string.show_details),
                    tint = colors.onSurfaceVariant)
            }
            AnimatedVisibility(details) {
                Column(Modifier.padding(end = 8.dp, top = 8.dp, bottom = 8.dp)) {
                    Text(strings.getString(model.descriptionRes), style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
                    Spacer(Modifier.height(8.dp))
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Pill(if (model.streamed) strings.getString(R.string.expert_streaming) else strings.getString(R.string.in_memory_134), colors.surfaceContainerHighest, colors.onSurface)
                        if (model.projector != null) Pill(strings.getString(R.string.images_135), colors.surfaceContainerHighest, colors.onSurface)
                        Pill(model.license, colors.surfaceContainerHighest, colors.onSurfaceVariant)
                    }
                    if (model.streamed && installed) MemoryNeed(model)
                    if (model.streamed && !installed && partial == 0L) {
                        Spacer(Modifier.height(6.dp))
                        Text(strings.getString(R.string.downloads_over_wi_fi_only_and_continues_in_the_background),
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
                Text(if (installed) strings.getString(R.string.delete_model) else strings.getString(R.string.discard_download), color = colors.error)
            }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(if (installed) strings.getString(R.string.delete_1_s, model.name) else strings.getString(R.string.discard_the_download)) },
            text = { Text(strings.getString(R.string.this_frees_up_to_1_s_you_can_download_it_again_later, formatBytes(model.sizeBytes))) },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    if (engine.modelId == model.id) EngineController.unload()
                    if (installed) ChatStore.deleteSessionsOf(model.sha256)
                    Downloads.delete(context, model.artifact) { refresh++ }
                }) { Text(strings.getString(R.string.delete), color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(strings.getString(R.string.keep)) } },
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
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    var options by remember { mutableStateOf(false) }
    Card(
        colors = CardDefaults.cardColors(containerColor = colors.surfaceContainerLow.copy(alpha = 0.8f)),
        border = BorderStroke(1.dp, BrandGradient),
        shape = RoundedCornerShape(24.dp),
    ) {
        Row(Modifier.padding(start = 16.dp, end = 8.dp, top = 10.dp, bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(engine.modelName ?: strings.getString(R.string.local_model), style = MaterialTheme.typography.titleSmall)
                Text(strings.getString(R.string.opened_from_this_phone_s_files), style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
            }
            Pill(strings.getString(R.string.active), colors.primaryContainer, colors.onPrimaryContainer)
            IconButton(onClick = { options = true }) {
                Icon(Icons.Outlined.MoreVert, contentDescription = strings.getString(R.string.options_for_the_local_model), tint = colors.onSurfaceVariant)
            }
        }
    }
    if (options) ModelOptionsSheet(engine.modelName ?: strings.getString(R.string.local_model), strings.getString(R.string.opened_from_this_phone_s_files), { options = false }) {
        SteeringSection(engine.modelSha256, engine, loaded = true)
    }
}

/**
 * The directional steering vector of one exact model file (by fingerprint): imported here and offered in that
 * model's chats while it is loaded.
 */
@Composable
private fun SteeringSection(modelHash: String?, engine: EngineState, loaded: Boolean) {
    val strings = LocalContext.current.resources
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
            HelpLabel(
                strings.getString(R.string.directional_steering),
                strings.getString(R.string.import_a_vector_made_for_this_exact_file_to_steer_its_answers_chats_offer_i) + "\n\n" +
                    strings.getString(R.string.select_the_vector_gguf_and_its_matching_gguf_json_together) + "\n\n" +
                    if (loaded) strings.getString(R.string.turn_it_on_per_chat_with_the_configuration_button)
                    else strings.getString(R.string.available_in_chats_once_this_model_is_loaded),
                summary = strings.getString(R.string.steering_summary),
                style = MaterialTheme.typography.titleSmall,
            )
            if (modelHash == null) Text(strings.getString(R.string.verify_this_file_s_fingerprint_to_import_a_vector_for_it),
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
            else if (loaded && engine.steeringProfile == null && engine.steeringNote != null) Text(engine.steeringNote,
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
            else if (current != null) Text(
                strings.getString(R.string.vector_imported_layer_1_d, current.layer) +
                    (current.direction.takeIf { it.isNotEmpty() }?.let { " · ${it.replace("_minus_", " − ").replace('_', ' ')}" } ?: ""),
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
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
            TextButton(onClick = { EngineController.stop() }) { Text(strings.getString(R.string.cancel)) }
        }
        modelHash == null -> Column {
            HelpLabel(strings.getString(R.string.verify_local_model),
                strings.getString(R.string.reads_and_reloads_the_model_once_to_check_its_exact_fingerprint),
                summary = strings.getString(R.string.verify_model_summary))
            OutlinedButton(onClick = { EngineController.verifySteeringModel(context) }, enabled = engine.phase == Phase.Ready) {
                Text(strings.getString(R.string.verify_local_model))
            }
        }
        else -> Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = { importer.launch(arrayOf("*/*")) }, enabled = !engine.busy) {
                    Text(if (current == null) strings.getString(R.string.import_vector_json) else strings.getString(R.string.replace_vector))
                }
                if (current != null) {
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = { confirmRemove = true }, enabled = !engine.busy) { Text(strings.getString(R.string.remove)) }
                }
            }
        }
    }
    if (confirmRemove && modelHash != null) {
        AlertDialog(
            onDismissRequest = { confirmRemove = false },
            title = { Text(strings.getString(R.string.remove_the_steering_vector)) },
            text = { Text(strings.getString(R.string.chats_of_this_model_stop_offering_steering_the_vector_can_be_imported_again)) },
            confirmButton = {
                TextButton(onClick = {
                    confirmRemove = false
                    EngineController.removeSteering(context, modelHash)
                }) { Text(strings.getString(R.string.remove), color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmRemove = false }) { Text(strings.getString(R.string.keep)) } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

/** Image input: the model's vision projector, downloaded and removed on its own. */
@Composable
private fun ProjectorSection(model: CatalogModel, projector: Artifact, engine: EngineState) {
    val strings = LocalContext.current.resources
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
            HelpLabel(
                strings.getString(R.string.image_input),
                if (model.streamed) strings.getString(R.string.lets_this_model_read_photos_screenshots_and_scanned_documents_while_it_read,
                    formatBytes(projector.sizeBytes + (128L shl 20)))
                else strings.getString(R.string.lets_this_model_read_photos_screenshots_and_scanned_documents_while_it_read_219,
                    formatBytes(projector.sizeBytes + (512L shl 20))),
                summary = strings.getString(R.string.image_input_summary),
                style = MaterialTheme.typography.titleSmall,
            )
            if (installed) Text(
                if (loaded && !engine.vision) strings.getString(R.string.downloaded_reload_the_model_to_read_images)
                else strings.getString(R.string.ready_photos_and_scanned_pdfs_can_be_attached),
                style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
        }
    }
    Spacer(Modifier.height(10.dp))
    when {
        transferring -> {
            DownloadProgress(projector, work)
            Spacer(Modifier.height(4.dp))
            TextButton(onClick = { Downloads.pause(context, projector) }) { Text(strings.getString(R.string.pause)) }
        }
        installed -> Row(verticalAlignment = Alignment.CenterVertically) {
            // Loads attach a downloaded projector on their own; a model loaded before the download needs one reload.
            if (loaded && !engine.vision) {
                OutlinedButton(onClick = { EngineController.load(context, model) }, enabled = engine.phase == Phase.Ready) {
                    Text(strings.getString(R.string.reload_with_images))
                }
                Spacer(Modifier.width(8.dp))
            }
            TextButton(onClick = { confirmDelete = true }, enabled = engine.phase != Phase.Generating) {
                Icon(Icons.Outlined.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text(strings.getString(R.string.remove_image_support))
            }
        }
        else -> Column(Modifier.fillMaxWidth()) {
            if (state == WorkInfo.State.FAILED) {
                Text(work?.outputData?.getString(Downloads.KEY_ERROR) ?: strings.getString(R.string.download_failed),
                    style = MaterialTheme.typography.bodySmall, color = colors.error)
                Spacer(Modifier.height(6.dp))
            }
            PremiumOutlinedButton(onClick = { Downloads.start(context, projector) },
                modifier = Modifier.align(Alignment.End), compact = true) {
                Icon(Icons.Rounded.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text(if (partial > 0) strings.getString(R.string.resume_1_s_of_2_s, formatBytes(partial), formatBytes(projector.sizeBytes))
                    else strings.getString(R.string.add_image_support_1_s, formatBytes(projector.sizeBytes)))
            }
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(strings.getString(R.string.remove_image_support_160)) },
            text = { Text(strings.getString(R.string.this_frees_1_s_models_that_share_it_lose_image_input_until_it_is_downloaded, formatBytes(projector.sizeBytes))) },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    // The engine reads the projector on every image request; release a model loaded with it first.
                    if (Catalog.byId(engine.modelId)?.projector?.id == projector.id && engine.vision) EngineController.unload()
                    Downloads.delete(context, projector) { refresh++ }
                }) { Text(strings.getString(R.string.remove), color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(strings.getString(R.string.keep)) } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

@Composable
private fun DownloadProgress(model: Artifact, work: WorkInfo?) {
    val strings = LocalContext.current.resources
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
        waiting -> if (model.unmeteredOnly) strings.getString(R.string.waiting_for_wi_fi) else strings.getString(R.string.waiting_for_a_connection)
        verifying -> strings.getString(R.string.checking_the_partial_download_1_s, formatBytes(done))
        else -> buildString {
            append(strings.getString(R.string.value_1_s_of_2_s, formatBytes(done), formatBytes(total)))
            if (speed > 0) {
                append(" · ${formatBytes(speed)}/s")
                val minutes = (total - done) / speed / 60
                append(if (minutes >= 1) strings.getString(R.string.value_1_d_min_left, minutes) else strings.getString(R.string.under_a_minute_left))
            }
        }
    }
    Text(text, style = MaterialTheme.typography.bodySmall, color = colors.onSurfaceVariant)
}

@Composable
fun ThinProgress(progress: Float?) = GradientProgress(progress)

@Composable
fun Pill(text: String, container: androidx.compose.ui.graphics.Color, content: androidx.compose.ui.graphics.Color) {
    Surface(color = container, shape = RoundedCornerShape(50)) {
        Text(text, style = MaterialTheme.typography.labelSmall, color = content,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp))
    }
}
