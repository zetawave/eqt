package org.equity.app.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.outlined.Inventory2
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import org.equity.app.EngineController
import org.equity.app.Phase

enum class Tab(val label: String, val icon: ImageVector) {
    Chat("Chat", Icons.AutoMirrored.Outlined.Chat),
    Models("Models", Icons.Outlined.Inventory2),
    Settings("Settings", Icons.Outlined.Tune),
}

@Composable
fun EquityApp() {
    val engine by EngineController.state.collectAsStateWithLifecycle()
    var tab by rememberSaveable { mutableStateOf(if (engine.phase == Phase.Empty) Tab.Models else Tab.Chat) }
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(engine.error) {
        engine.error?.let {
            EngineController.clearError()
            snackbar.showSnackbar(it)
        }
    }
    // A finished load lands in the conversation.
    LaunchedEffect(engine.phase, engine.modelId) {
        if (engine.phase == Phase.Ready && engine.messages.isEmpty() && tab == Tab.Models) tab = Tab.Chat
    }
    val colors = MaterialTheme.colorScheme
    Scaffold(
        containerColor = colors.background,
        snackbarHost = { SnackbarHost(snackbar) },
        bottomBar = {
            NavigationBar(containerColor = colors.surfaceContainerLow, tonalElevation = 0.dp) {
                Tab.entries.forEach { item ->
                    NavigationBarItem(
                        selected = tab == item,
                        onClick = { tab = item },
                        icon = { Icon(item.icon, contentDescription = null) },
                        label = { Text(item.label, style = MaterialTheme.typography.labelMedium) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = colors.onPrimaryContainer,
                            selectedTextColor = colors.onSurface,
                            indicatorColor = colors.primaryContainer,
                            unselectedIconColor = colors.onSurfaceVariant,
                            unselectedTextColor = colors.onSurfaceVariant,
                        ),
                    )
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding).consumeWindowInsets(padding)) {
            when (tab) {
                Tab.Chat -> ChatScreen(engine, onOpenModels = { tab = Tab.Models })
                Tab.Models -> ModelsScreen(engine, onOpenChat = { tab = Tab.Chat })
                Tab.Settings -> SettingsScreen(engine)
            }
        }
    }
}
