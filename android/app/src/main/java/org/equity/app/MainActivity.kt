package org.equity.app

import android.content.ComponentCallbacks2
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import org.equity.app.ui.EquityApp
import org.equity.app.ui.EquityTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        SettingsStore.init(this)
        setContent { EquityTheme { EquityApp() } }
    }

    // No background inference: leaving the app stops the current answer.
    override fun onStop() {
        if (EngineController.state.value.phase == Phase.Generating) EngineController.stop()
        super.onStop()
    }

    // Android 14+ only reports UI_HIDDEN and BACKGROUND. A multi-GiB streamed model is released once the app is
    // in the background so the rest of the phone stays responsive; a small resident model is kept.
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        val model = Catalog.byId(EngineController.state.value.modelId)
        val large = model?.streamed ?: true
        @Suppress("DEPRECATION")
        val critical = level >= ComponentCallbacks2.TRIM_MEMORY_MODERATE || level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL
        if (critical || (large && level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND)) EngineController.memoryPressure()
    }
}
