package org.equity.app.ui

import android.text.format.DateUtils
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import org.equity.app.ChatStore
import org.equity.app.ChatSummary
import org.equity.app.ChatsState
import org.equity.app.EngineController
import org.equity.app.EngineState

@Composable
fun ChatDrawer(chats: ChatsState, engine: EngineState, onSelect: (String) -> Unit, onNewChat: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    var deleting by remember { mutableStateOf<ChatSummary?>(null) }
    ModalDrawerSheet(drawerContainerColor = colors.surfaceContainerLow, drawerShape = RoundedCornerShape(topEnd = 24.dp, bottomEnd = 24.dp)) {
        Column(Modifier.fillMaxSize().statusBarsPadding().padding(top = 16.dp)) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("Chats", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
                FilledTonalButton(onClick = onNewChat, enabled = chats.currentId != null) {
                    Icon(Icons.Rounded.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("New chat")
                }
            }
            Spacer(Modifier.height(12.dp))
            if (chats.summaries.isEmpty()) {
                Text("No conversations yet. They are saved here as you chat.",
                    style = MaterialTheme.typography.bodyMedium, color = colors.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp))
            }
            LazyColumn(
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                items(chats.summaries, key = { it.id }) { chat ->
                    ChatRow(chat, selected = chat.id == chats.currentId,
                        answering = engine.generation?.chatId == chat.id,
                        onClick = { onSelect(chat.id) }, onDelete = { deleting = chat })
                }
            }
            Row(Modifier.fillMaxWidth().padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.Lock, contentDescription = null, tint = colors.onSurfaceVariant, modifier = Modifier.size(12.dp))
                Spacer(Modifier.width(6.dp))
                Text("Chats and attachments are stored only on this phone.", style = MaterialTheme.typography.labelSmall,
                    color = colors.onSurfaceVariant)
            }
        }
    }
    deleting?.let { chat ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete this chat?") },
            text = { Text("\"${chat.title}\" and its attachments will be removed from this phone.") },
            confirmButton = {
                TextButton(onClick = {
                    deleting = null
                    if (engine.generation?.chatId == chat.id) EngineController.stop()
                    ChatStore.delete(chat.id)
                }) { Text("Delete", color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Keep") } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

@Composable
private fun ChatRow(chat: ChatSummary, selected: Boolean, answering: Boolean, onClick: () -> Unit, onDelete: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    Surface(
        color = if (selected) colors.secondaryContainer else colors.surfaceContainerLow,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
    ) {
        Row(Modifier.padding(start = 16.dp, top = 8.dp, bottom = 8.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(chat.title, style = MaterialTheme.typography.bodyLarge, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    color = if (selected) colors.onSecondaryContainer else colors.onSurface)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (answering) {
                        StatusDot(org.equity.app.Phase.Generating)
                        Spacer(Modifier.width(6.dp))
                    }
                    Text(
                        if (answering) "Answering…"
                        else DateUtils.getRelativeTimeSpanString(chat.updatedAt, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS).toString(),
                        style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                    )
                }
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Outlined.Delete, contentDescription = "Delete chat", tint = colors.onSurfaceVariant, modifier = Modifier.size(20.dp))
            }
        }
    }
}
