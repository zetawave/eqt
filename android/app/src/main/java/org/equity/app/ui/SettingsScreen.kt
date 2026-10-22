package org.equity.app.ui

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Language
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.core.net.toUri
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.equity.app.ANSWER_UNLIMITED
import org.equity.app.BuildConfigInfo
import org.equity.app.CONTEXT_AUTO
import org.equity.app.CONTEXT_MAX
import org.equity.app.ChatStore
import org.equity.app.EngineController
import org.equity.app.EngineState
import org.equity.app.R
import org.equity.app.SettingsStore
import org.json.JSONObject
import java.util.Locale
import kotlin.math.roundToInt

private const val MIN_ANSWER_TOKENS = 256
// The slider's end; beyond it, "No limit" lets the answer run as long as the model and the context window allow.
private const val MAX_ANSWER_TOKENS = 32_768
private const val MAX_TEMPERATURE = 1.2f
private const val WEBSITE_URL = "https://zetawave.github.io/eqt/"
private const val REPOSITORY_URL = "https://github.com/zetawave/eqt"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(engine: EngineState) {
    val strings = LocalContext.current.resources
    val context = LocalContext.current
    val settings by SettingsStore.state.collectAsStateWithLifecycle()
    val chats by ChatStore.state.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    var confirmDeleteAll by remember { mutableStateOf(false) }
    var licenses by remember { mutableStateOf<List<String>?>(null) }
    var licenseText by remember { mutableStateOf<Pair<String, String>?>(null) }
    var exported by remember { mutableStateOf<String?>(null) }
    val exporter = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        val text = engine.lastResult
        if (uri != null && text != null) scope.launch {
            exported = withContext(Dispatchers.IO) {
                runCatching { context.contentResolver.openOutputStream(uri)?.use { it.write(text.toByteArray()) }; strings.getString(R.string.exported) }
                    .getOrElse { it.message ?: strings.getString(R.string.export_failed) }
            }
        }
    }
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding()
            .padding(start = 20.dp, end = 20.dp, top = 20.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(strings.getString(R.string.settings), style = MaterialTheme.typography.headlineMedium)

        Section(strings.getString(R.string.conversation)) {
            OutlinedTextField(
                value = settings.systemPrompt,
                onValueChange = { value -> SettingsStore.update(context) { it.copy(systemPrompt = value) } },
                label = { Text(strings.getString(R.string.system_prompt)) },
                minLines = 3,
                maxLines = 6,
                textStyle = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth(),
            )
            val unlimited = settings.maxTokens == ANSWER_UNLIMITED
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(strings.getString(R.string.answer_length_without_limit), style = MaterialTheme.typography.bodyMedium)
                    Text(
                        if (unlimited) strings.getString(R.string.answers_run_until_the_model_ends_them_or_the_context_window_is_full)
                        else strings.getString(R.string.answers_stop_at_the_length_below),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(checked = unlimited, onCheckedChange = { on ->
                    SettingsStore.update(context) { it.copy(maxTokens = if (on) ANSWER_UNLIMITED else 2048) }
                })
            }
            if (!unlimited) SliderField(
                label = strings.getString(R.string.maximum_answer_length),
                value = settings.maxTokens.toFloat(),
                onValueChange = { value -> SettingsStore.update(context) { it.copy(maxTokens = value.roundToInt()) } },
                range = MIN_ANSWER_TOKENS.toFloat()..MAX_ANSWER_TOKENS.toFloat(),
                step = 256f,
                format = { it.roundToInt().toString() },
                integer = true,
                hint = strings.getString(R.string.tokens_1_d_to_2_d_thinking_uses_part_of_them, MIN_ANSWER_TOKENS, MAX_ANSWER_TOKENS),
            )
            SliderField(
                label = strings.getString(R.string.creativity),
                value = settings.temperature,
                onValueChange = { value -> SettingsStore.update(context) { it.copy(temperature = value) } },
                range = 0f..MAX_TEMPERATURE,
                step = 0.1f,
                format = { formatDecimal(it, 2) },
                hint = if (settings.temperature == 0f) strings.getString(R.string.0_most_predictable) else strings.getString(R.string.higher_values_vary_answers_more),
            )
        }

        Section(strings.getString(R.string.performance), strings.getString(R.string.applies_the_next_time_a_model_is_loaded)) {
            Text(strings.getString(R.string.processor), style = MaterialTheme.typography.labelLarge)
            Choice(listOf(false, true), settings.gpu, { if (it) "CPU + GPU" else "CPU" }) { value ->
                SettingsStore.update(context) { it.copy(gpu = value) }
            }
            Text(strings.getString(R.string.cpu_gpu_reads_long_prompts_chat_history_and_images_faster_on_the_gpu_and_us),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(strings.getString(R.string.cpu_threads), style = MaterialTheme.typography.labelLarge)
            Choice(listOf(2, 4, 6), settings.threads, { "$it" }) { value -> SettingsStore.update(context) { it.copy(threads = value) } }
            Text(strings.getString(R.string.context_window), style = MaterialTheme.typography.labelLarge)
            Choice(listOf(CONTEXT_AUTO, 2048, 4096, 8192, CONTEXT_MAX), settings.context, { value ->
                when (value) {
                    CONTEXT_AUTO -> strings.getString(R.string.auto)
                    CONTEXT_MAX -> strings.getString(R.string.max)
                    else -> "${value / 1024}K"
                }
            }) { value -> SettingsStore.update(context) { it.copy(context = value) } }
            Text(
                when (settings.context) {
                    CONTEXT_AUTO -> strings.getString(R.string.auto_sizes_the_window_from_free_memory_at_load_without_taking_the_expert_ca)
                    CONTEXT_MAX -> strings.getString(R.string.max_loads_the_model_s_full_trained_length_or_as_much_as_free_memory_allows)
                    else -> strings.getString(R.string.starting_size)
                } + strings.getString(R.string.the_window_grows_for_longer_messages_and_attachments_when_memory_allows_the),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(strings.getString(R.string.expert_cache_large_model), style = MaterialTheme.typography.labelLarge)
            Choice(listOf(0, 1024, 2048, 3072), settings.expertCacheMiB, { if (it == 0) strings.getString(R.string.auto) else "${it / 1024} GB" }) { value ->
                SettingsStore.update(context) { it.copy(expertCacheMiB = value) }
            }
            Text(strings.getString(R.string.auto_sizes_the_cache_from_free_memory_a_larger_cache_reads_storage_less_oft),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        Section(strings.getString(R.string.diagnostics)) {
            Labeled(strings.getString(R.string.engine), engine.configuration ?: strings.getString(R.string.no_model_loaded))
            val result = engine.lastResult?.let { runCatching { JSONObject(it) }.getOrNull() }
            if (result != null) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Labeled(strings.getString(R.string.generation), strings.getString(R.string.1_2f_tokens_s, result.optDouble("decode_tokens_per_second")))
                Labeled(strings.getString(R.string.prompt), strings.getString(R.string.1_d_tokens_in_2_1f_s, result.optInt("prompt_tokens"), result.optDouble("prefill_ms") / 1000))
                result.optJSONObject("expert_cache")?.let { cache ->
                    val hits = cache.optLong("hits")
                    val total = hits + cache.optLong("misses")
                    if (total > 0) Labeled(strings.getString(R.string.expert_cache_hits), "%.0f%%".format(Locale.US, hits * 100.0 / total))
                }
                result.optJSONObject("memory_after")?.optLong("peak_rss_bytes")?.takeIf { it > 0 }?.let {
                    Labeled(strings.getString(R.string.peak_memory), "${it shr 20} MiB")
                }
                OutlinedButton(onClick = { exporter.launch("equity-run.json") }) { Text(strings.getString(R.string.export_last_run_json)) }
                exported?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            } else {
                Text(strings.getString(R.string.run_a_conversation_to_see_speed_and_memory_details), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        Section(strings.getString(R.string.privacy)) {
            Text(
                strings.getString(R.string.equity_runs_models_entirely_on_this_phone_prompts_answers_and_conversations),
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedButton(
                onClick = { confirmDeleteAll = true },
                enabled = chats.summaries.isNotEmpty(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
            ) {
                Icon(Icons.Outlined.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text(if (chats.summaries.isEmpty()) strings.getString(R.string.no_chats_to_delete) else strings.getString(R.string.delete_all_chats))
            }
        }

        Section(strings.getString(R.string.about)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Logo(28.dp)
                Spacer(Modifier.weight(1f))
                Text(strings.getString(R.string.version_1_s, BuildConfigInfo.VERSION), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text(strings.getString(R.string.developed_by_zetawave), style = MaterialTheme.typography.bodyMedium)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { openLink(context, WEBSITE_URL) }, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Outlined.Language, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(strings.getString(R.string.website))
                }
                OutlinedButton(onClick = { openLink(context, REPOSITORY_URL) }, modifier = Modifier.weight(1f)) {
                    Icon(painterResource(R.drawable.ic_github), contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("GitHub")
                }
            }
            Text(strings.getString(R.string.built_on_llama_cpp_and_ggml_model_weights_belong_to_their_publishers_under),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            TextButton(onClick = {
                licenses = context.assets.list("")?.filter { it.endsWith(".txt") }?.sorted().orEmpty()
            }) { Text(strings.getString(R.string.open_source_licenses)) }
        }
    }

    if (confirmDeleteAll) {
        val count = chats.summaries.size
        AlertDialog(
            onDismissRequest = { confirmDeleteAll = false },
            title = { Text(strings.getString(R.string.delete_all_chats_90)) },
            text = {
                Text(
                    (if (count == 1) strings.getString(R.string.the_chat_with_its_attachments_and_saved_state_is)
                    else strings.getString(R.string.all_1_d_chats_with_their_attachments_and_saved_states_are, count)) +
                        strings.getString(R.string.removed_from_this_phone_this_cannot_be_undone) +
                        if (engine.generation != null) strings.getString(R.string.the_answer_being_written_is_stopped) else "",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (engine.generation != null) EngineController.stop()
                        ChatStore.deleteAll()
                        confirmDeleteAll = false
                    },
                    colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error),
                ) { Text(strings.getString(R.string.delete_all)) }
            },
            dismissButton = { TextButton(onClick = { confirmDeleteAll = false }) { Text(strings.getString(R.string.cancel)) } },
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        )
    }
    licenses?.let { files ->
        AlertDialog(
            onDismissRequest = { licenses = null },
            title = { Text(strings.getString(R.string.licenses)) },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    files.forEach { file ->
                        TextButton(onClick = {
                            licenseText = file to context.assets.open(file).bufferedReader().use { it.readText() }
                        }) { Text(file.removeSuffix(".txt")) }
                    }
                }
            },
            confirmButton = { TextButton(onClick = { licenses = null }) { Text(strings.getString(R.string.close)) } },
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        )
    }
    licenseText?.let { (name, text) ->
        AlertDialog(
            onDismissRequest = { licenseText = null },
            title = { Text(name.removeSuffix(".txt")) },
            text = {
                Text(text, fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp,
                    modifier = Modifier.heightIn(max = 480.dp).verticalScroll(rememberScrollState()))
            },
            confirmButton = { TextButton(onClick = { licenseText = null }) { Text(strings.getString(R.string.close)) } },
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        )
    }
}

/** Opens a web page in the browser; the app itself never loads it. */
private fun openLink(context: Context, url: String) {
    val strings = context.resources
    try {
        context.startActivity(Intent(Intent.ACTION_VIEW, url.toUri()))
    } catch (_: ActivityNotFoundException) {
        Toast.makeText(context, strings.getString(R.string.no_app_on_this_phone_can_open_1_s, url), Toast.LENGTH_LONG).show()
    }
}

@Composable
private fun Section(title: String, subtitle: String? = null, content: @Composable ColumnScope.() -> Unit) {
    val strings = LocalContext.current.resources
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, shape = RoundedCornerShape(24.dp)) {
        Column(Modifier.fillMaxWidth().padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Column {
                Text(title, style = MaterialTheme.typography.titleMedium)
                subtitle?.let {
                    Spacer(Modifier.height(2.dp))
                    Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            content()
        }
    }
}

@Composable
private fun Labeled(label: String, value: String) {
    val strings = LocalContext.current.resources
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> Choice(options: List<T>, selected: T, label: (T) -> String, onSelect: (T) -> Unit) {
    val strings = LocalContext.current.resources
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
        options.forEachIndexed { index, option ->
            SegmentedButton(
                selected = option == selected,
                onClick = { onSelect(option) },
                shape = SegmentedButtonDefaults.itemShape(index, options.size),
                label = { Text(label(option)) },
            )
        }
    }
}
