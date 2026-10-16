package org.equity.app

import android.app.Activity
import android.app.Application
import android.content.ComponentCallbacks2
import android.os.Bundle
import kotlinx.coroutines.flow.MutableStateFlow

/** Whether any Equity activity is started (on screen); answers finished otherwise get a notification. */
object AppVisibility {
    @Volatile var visible = false
        internal set
}

/** A chat to show, requested by a notification tap; the UI consumes it. */
object AppNavigation {
    val openChat = MutableStateFlow<String?>(null)
}

class EquityApplication : Application() {
    private var started = 0

    override fun onCreate() {
        super.onCreate()
        SettingsStore.init(this)
        ChatStore.init(this)
        Notifications.ensureChannels(this)
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityStarted(activity: Activity) {
                AppVisibility.visible = ++started > 0
            }
            override fun onActivityStopped(activity: Activity) {
                AppVisibility.visible = --started > 0
            }
            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) = Unit
            override fun onActivityResumed(activity: Activity) = Unit
            override fun onActivityPaused(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit
        })
    }

    // Android 14+ only reports UI_HIDDEN and BACKGROUND. A multi-GiB streamed model is released once the app is
    // in the background so the rest of the phone stays responsive; a small resident model is kept. Work in
    // progress (an answer or a load, held by InferenceService) is only interrupted when memory is critical.
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        val engine = EngineController.state.value
        val large = Catalog.byId(engine.modelId)?.streamed ?: true
        @Suppress("DEPRECATION")
        val critical = level >= ComponentCallbacks2.TRIM_MEMORY_MODERATE || level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL
        when {
            critical -> EngineController.memoryPressure("Android was out of memory, so the model was unloaded.")
            engine.busy -> Unit
            large && level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND ->
                EngineController.memoryPressure("The model was unloaded while Equity was in the background, to keep the phone responsive.")
        }
    }
}
