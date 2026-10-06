package org.equity.app

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import org.json.JSONObject

class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var configuration: TextView
    private lateinit var conversation: TextView
    private lateinit var prompt: EditText
    private lateinit var system: EditText
    private lateinit var contextSize: EditText
    private lateinit var threads: EditText
    private lateinit var budget: EditText
    private lateinit var output: EditText
    private lateinit var temperature: EditText
    private lateinit var progress: ProgressBar
    private val idleButtons = mutableListOf<Button>()
    private var exportText: String? = null

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
        java.io.File(getExternalFilesDir(null), "models").mkdirs()
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(28, 16, 28, 16) }
        layout.setOnApplyWindowInsetsListener { view, insets ->
            view.setPadding(28, insets.systemWindowInsetTop + 16, 28, insets.systemWindowInsetBottom + 16)
            insets
        }
        fun label(text: String) = TextView(this).also { it.text = text; layout.addView(it) }
        label("Equity · local CPU research build").textSize = 22f
        status = label(EngineController.status)
        configuration = label(EngineController.configuration)
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { isIndeterminate = true }
        layout.addView(progress)
        val modelRow = LinearLayout(this)
        layout.addView(modelRow)
        fun button(parent: LinearLayout, title: String, idle: Boolean = true, action: () -> Unit): Button {
            return Button(this).also {
                it.text = title
                it.setOnClickListener { attempt(action) }
                parent.addView(it, LinearLayout.LayoutParams(0, -2, 1f))
                if (idle) idleButtons += it
            }
        }
        button(modelRow, "Open GGUF") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }, 1)
        }
        button(modelRow, "Staged model") {
            EngineController.prepareLoad()
            EngineController.load(this, null, loadOptions())
        }
        button(modelRow, "Unload") { EngineController.unload() }

        val technical = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = View.GONE }
        Button(this).also {
            it.text = "Technical settings"
            it.setOnClickListener { technical.visibility = if (technical.visibility == View.GONE) View.VISIBLE else View.GONE }
            layout.addView(it)
        }
        val settingsScroll = ScrollView(this).apply { addView(technical) }
        layout.addView(settingsScroll, LinearLayout.LayoutParams(-1, -2).apply { height = -2 })
        fun field(hint: String, value: String): EditText = EditText(this).also {
            it.hint = hint; it.setText(value); it.contentDescription = hint; technical.addView(it)
        }
        system = field("System prompt", "You are a concise assistant. Follow the requested language.")
        contextSize = field("Context (reload to apply)", "2048")
        threads = field("Threads (reload to apply)", "4")
        budget = field("Resident admission MiB (reload to apply)", "2048")
        output = field("Maximum output tokens", "128")
        temperature = field("Temperature (0 = greedy)", "0")
        TextView(this).also {
            it.text = "CPU only · mmap · no expert cache · no MTP · no steering. Model selection uses a seekable descriptor without a copy. Settings for context, threads and memory apply on reload."
            technical.addView(it)
        }
        val scroll = ScrollView(this)
        conversation = TextView(this).apply { textSize = 16f; setTextIsSelectable(true) }
        scroll.addView(conversation)
        layout.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        prompt = EditText(this).apply { hint = "Message (stays on this device)"; minLines = 1; maxLines = 3 }
        layout.addView(prompt)
        val actions = LinearLayout(this)
        layout.addView(actions)
        button(actions, "Send") {
            require(EngineController.loaded) { "Load a model first" }
            require(prompt.text.isNotBlank()) { "Enter a message" }
            EngineController.generate(this, prompt.text.toString(), system.text.toString(), output.text.toString().toInt(), temperature.text.toString().toDouble(), false)
            prompt.text.clear()
        }
        button(actions, "Stop", false) { EngineController.stop() }
        button(actions, "New chat") { EngineController.newChat() }
        val benchRow = LinearLayout(this)
        layout.addView(benchRow)
        button(benchRow, "Benchmark") {
            EngineController.generate(this, "Explain in Italian how RAM differs from storage.", system.text.toString(), 128, 0.0, true)
        }
        button(benchRow, "Export result") {
            exportText = EngineController.lastResult
            require(exportText != null) { "Run a generation or benchmark first" }
            AlertDialog.Builder(this).setMessage("Export timings, configuration and generated answer to a local JSON file?")
                .setPositiveButton("Export") { _, _ ->
                    startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        type = "application/json"; addCategory(Intent.CATEGORY_OPENABLE); putExtra(Intent.EXTRA_TITLE, "equity-result.json")
                    }, 2)
                }.setNegativeButton("Cancel", null).show()
        }
        setContentView(layout)
    }

    private fun loadOptions() = JSONObject().put("context", contextSize.text.toString().toInt())
        .put("threads", threads.text.toString().toInt()).put("memory_budget_mib", budget.text.toString().toInt())
        .put("batch", 128).put("mmap", true)

    private fun attempt(action: () -> Unit) {
        try { action() } catch (error: Exception) { status.text = error.message }
    }
    private fun refresh() {
        status.text = EngineController.status
        configuration.text = EngineController.configuration
        conversation.text = EngineController.transcript
        progress.visibility = if (EngineController.busy) View.VISIBLE else View.GONE
        idleButtons.forEach { it.isEnabled = !EngineController.busy }
    }
    override fun onStart() {
        super.onStart()
        EngineController.observer = { refresh() }
        refresh()
    }
    override fun onStop() {
        EngineController.observer = null
        if (EngineController.busy) EngineController.stop()
        super.onStop()
    }
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level == TRIM_MEMORY_RUNNING_LOW || level == TRIM_MEMORY_RUNNING_CRITICAL || level >= TRIM_MEMORY_BACKGROUND) {
            EngineController.memoryPressure()
        }
    }
    @Deprecated("Legacy activity result API is sufficient for this platform-only prototype")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        attempt {
            if (requestCode == 1) {
                EngineController.prepareLoad()
                EngineController.load(this, uri, loadOptions())
            } else if (requestCode == 2) {
                val snapshot = exportText ?: return@attempt
                Thread {
                    try {
                        contentResolver.openOutputStream(uri)?.use { it.write(snapshot.toByteArray()) }
                            ?: error("Cannot open export destination")
                        runOnUiThread { status.text = "Result exported" }
                    } catch (error: Exception) { runOnUiThread { status.text = error.message } }
                }.start()
            }
        }
    }
}
