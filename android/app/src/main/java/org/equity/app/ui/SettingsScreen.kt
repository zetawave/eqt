package org.equity.app.ui

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
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.equity.app.BuildConfigInfo
import org.equity.app.EngineState
import org.equity.app.SettingsStore
import org.json.JSONObject
import java.util.Locale
import kotlin.math.roundToInt

private val Lengths = listOf(128, 256, 512, 1024, 2048)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(engine: EngineState) {
    val context = LocalContext.current
    val settings by SettingsStore.state.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    var licenses by remember { mutableStateOf<List<String>?>(null) }
    var licenseText by remember { mutableStateOf<Pair<String, String>?>(null) }
    var exported by remember { mutableStateOf<String?>(null) }
    val exporter = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        val text = engine.lastResult
        if (uri != null && text != null) scope.launch {
            exported = withContext(Dispatchers.IO) {
                runCatching { context.contentResolver.openOutputStream(uri)?.use { it.write(text.toByteArray()) }; "Exported" }
                    .getOrElse { it.message ?: "Export failed" }
            }
        }
    }
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding()
            .padding(start = 20.dp, end = 20.dp, top = 20.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium)

        Section("Conversation") {
            OutlinedTextField(
                value = settings.systemPrompt,
                onValueChange = { value -> SettingsStore.update(context) { it.copy(systemPrompt = value) } },
                label = { Text("System prompt") },
                minLines = 3,
                maxLines = 6,
                textStyle = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth(),
            )
            val lengthIndex = Lengths.indexOf(settings.maxTokens).coerceAtLeast(0)
            Labeled("Maximum answer length", "${settings.maxTokens} tokens")
            Slider(
                value = lengthIndex.toFloat(),
                onValueChange = { index -> SettingsStore.update(context) { it.copy(maxTokens = Lengths[index.roundToInt()]) } },
                valueRange = 0f..(Lengths.size - 1).toFloat(),
                steps = Lengths.size - 2,
            )
            Labeled("Creativity", if (settings.temperature == 0f) "0.0 · most predictable" else "%.1f".format(Locale.US, settings.temperature))
            Slider(
                value = settings.temperature,
                onValueChange = { value -> SettingsStore.update(context) { it.copy(temperature = (value * 10).roundToInt() / 10f) } },
                valueRange = 0f..1.2f,
                steps = 11,
            )
        }

        Section("Performance", "Applies the next time a model is loaded.") {
            Text("CPU threads", style = MaterialTheme.typography.labelLarge)
            Choice(listOf(2, 4, 6), settings.threads, { "$it" }) { value -> SettingsStore.update(context) { it.copy(threads = value) } }
            Text("Context window", style = MaterialTheme.typography.labelLarge)
            Choice(listOf(2048, 4096, 8192), settings.context, { "${it / 1024}K" }) { value -> SettingsStore.update(context) { it.copy(context = value) } }
            Text("Expert cache (large model)", style = MaterialTheme.typography.labelLarge)
            Choice(listOf(0, 1024, 2048, 3072), settings.expertCacheMiB, { if (it == 0) "Auto" else "${it / 1024} GB" }) { value ->
                SettingsStore.update(context) { it.copy(expertCacheMiB = value) }
            }
            Text("Auto sizes the cache from free memory. A larger cache reads storage less often and is faster.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        Section("Diagnostics") {
            Labeled("Engine", engine.configuration ?: "No model loaded")
            val result = engine.lastResult?.let { runCatching { JSONObject(it) }.getOrNull() }
            if (result != null) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Labeled("Generation", "%.2f tokens/s".format(Locale.US, result.optDouble("decode_tokens_per_second")))
                Labeled("Prompt", "%d tokens in %.1f s".format(Locale.US, result.optInt("prompt_tokens"), result.optDouble("prefill_ms") / 1000))
                result.optJSONObject("expert_cache")?.let { cache ->
                    val hits = cache.optLong("hits")
                    val total = hits + cache.optLong("misses")
                    if (total > 0) Labeled("Expert cache hits", "%.0f%%".format(Locale.US, hits * 100.0 / total))
                }
                result.optJSONObject("memory_after")?.optLong("peak_rss_bytes")?.takeIf { it > 0 }?.let {
                    Labeled("Peak memory", "${it shr 20} MiB")
                }
                OutlinedButton(onClick = { exporter.launch("equity-run.json") }) { Text("Export last run (JSON)") }
                exported?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            } else {
                Text("Run a conversation to see speed and memory details.", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        Section("Privacy") {
            Text(
                "Equity runs models entirely on this phone. Prompts, answers and conversations are never uploaded, " +
                    "and there are no accounts, ads or analytics. The network is used only for model downloads you start.",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Section("About") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Logo(28.dp)
                Spacer(Modifier.weight(1f))
                Text("Version ${BuildConfigInfo.VERSION}", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text("Built on llama.cpp and GGML. Model weights belong to their publishers under their own licenses.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            TextButton(onClick = {
                licenses = context.assets.list("")?.filter { it.endsWith(".txt") }?.sorted().orEmpty()
            }) { Text("Open-source licenses") }
        }
    }

    licenses?.let { files ->
        AlertDialog(
            onDismissRequest = { licenses = null },
            title = { Text("Licenses") },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    files.forEach { file ->
                        TextButton(onClick = {
                            licenseText = file to context.assets.open(file).bufferedReader().use { it.readText() }
                        }) { Text(file.removeSuffix(".txt")) }
                    }
                }
            },
            confirmButton = { TextButton(onClick = { licenses = null }) { Text("Close") } },
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
            confirmButton = { TextButton(onClick = { licenseText = null }) { Text("Close") } },
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
        )
    }
}

@Composable
private fun Section(title: String, subtitle: String? = null, content: @Composable ColumnScope.() -> Unit) {
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
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> Choice(options: List<T>, selected: T, label: (T) -> String, onSelect: (T) -> Unit) {
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
