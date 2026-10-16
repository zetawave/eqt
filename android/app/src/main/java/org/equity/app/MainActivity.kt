package org.equity.app

import android.content.Intent
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
        if (savedInstanceState == null) route(intent)
        setContent { EquityTheme { EquityApp() } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        route(intent)
    }

    // Answers keep being written in the background (InferenceService); a notification tap opens their chat.
    private fun route(intent: Intent?) {
        val chat = intent?.getStringExtra(Notifications.EXTRA_CHAT) ?: return
        ChatStore.open(chat)
        AppNavigation.openChat.value = chat
    }
}
