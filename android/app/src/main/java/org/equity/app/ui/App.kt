package org.equity.app.ui

import androidx.compose.ui.platform.LocalContext
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.outlined.Inventory2
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.launch
import org.equity.app.R
import org.equity.app.AppNavigation
import org.equity.app.ChatStore
import org.equity.app.EngineController
import org.equity.app.Phase

enum class Tab(val label: Int, val icon: ImageVector) {
    Chat(R.string.chat, Icons.AutoMirrored.Outlined.Chat),
    Models(R.string.models, Icons.Outlined.Inventory2),
    Settings(R.string.settings, Icons.Outlined.Tune),
}

@Composable
fun EquityApp() {
    val strings = LocalContext.current.resources
    val engine by EngineController.state.collectAsStateWithLifecycle()
    val chats by ChatStore.state.collectAsStateWithLifecycle()
    val requested by AppNavigation.openChat.collectAsStateWithLifecycle()
    var tab by rememberSaveable {
        mutableStateOf(if (engine.phase == Phase.Empty && chats.summaries.isEmpty()) Tab.Models else Tab.Chat)
    }
    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(engine.error) {
        engine.error?.let {
            snackbar.showSnackbar(it)
            EngineController.clearError(it)
        }
    }
    // A finished load lands in the conversation.
    LaunchedEffect(engine.phase, engine.modelId) {
        if (engine.phase == Phase.Ready && engine.generation == null && tab == Tab.Models) tab = Tab.Chat
    }
    // A notification tap opens the chat whose answer it reports.
    LaunchedEffect(requested) {
        if (requested != null) {
            tab = Tab.Chat
            drawer.close()
            AppNavigation.openChat.value = null
        }
    }
    val colors = MaterialTheme.colorScheme
    ModalNavigationDrawer(
        drawerState = drawer,
        gesturesEnabled = tab == Tab.Chat || drawer.isOpen,
        drawerContent = {
            ChatDrawer(
                chats = chats,
                engine = engine,
                onSelect = { id ->
                    ChatStore.open(id)
                    tab = Tab.Chat
                    scope.launch { drawer.close() }
                },
                onNewChat = {
                    ChatStore.newChat()
                    tab = Tab.Chat
                    scope.launch { drawer.close() }
                },
            )
        },
    ) {
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
                            label = { Text(strings.getString(item.label), style = MaterialTheme.typography.labelMedium) },
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
            Box(Modifier.padding(padding).consumeWindowInsets(padding).screenBackground()) {
                when (tab) {
                    Tab.Chat -> ChatScreen(engine, chats, onOpenModels = { tab = Tab.Models },
                        onOpenChats = { scope.launch { drawer.open() } })
                    Tab.Models -> ModelsScreen(engine, onOpenChat = { tab = Tab.Chat })
                    Tab.Settings -> SettingsScreen(engine)
                }
            }
        }
    }
}
