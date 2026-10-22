package org.equity.app.ui

import androidx.compose.ui.platform.LocalContext
import android.text.format.DateUtils
import androidx.compose.foundation.clickable
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
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
import org.equity.app.R
import org.equity.app.ChatStore
import org.equity.app.ChatSummary
import org.equity.app.ChatsState
import org.equity.app.EngineController
import org.equity.app.EngineState

@Composable
fun ChatDrawer(chats: ChatsState, engine: EngineState, onSelect: (String) -> Unit, onNewChat: () -> Unit) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    var deleting by remember { mutableStateOf<ChatSummary?>(null) }
    ModalDrawerSheet(modifier = Modifier.screenBackground(), drawerShape = RoundedCornerShape(topEnd = 24.dp, bottomEnd = 24.dp)) {
        Column(Modifier.fillMaxSize().statusBarsPadding().padding(top = 16.dp)) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(strings.getString(R.string.chats), style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
                PremiumOutlinedButton(onClick = onNewChat, enabled = chats.currentId != null) {
                    Icon(Icons.Rounded.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text(strings.getString(R.string.new_chat), style = MaterialTheme.typography.labelLarge)
                }
            }
            Spacer(Modifier.height(12.dp))
            if (chats.summaries.isEmpty()) {
                Text(strings.getString(R.string.no_conversations_yet_they_are_saved_here_as_you_chat),
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
                Text(strings.getString(R.string.chats_and_attachments_are_stored_only_on_this_phone), style = MaterialTheme.typography.labelSmall,
                    color = colors.onSurfaceVariant)
            }
        }
    }
    deleting?.let { chat ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text(strings.getString(R.string.delete_this_chat)) },
            text = { Text(strings.getString(R.string.value_1_s_and_its_attachments_will_be_removed_from_this_phone, chat.title)) },
            confirmButton = {
                TextButton(onClick = {
                    deleting = null
                    if (engine.generation?.chatId == chat.id) EngineController.stop()
                    ChatStore.delete(chat.id)
                }) { Text(strings.getString(R.string.delete), color = colors.error) }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text(strings.getString(R.string.keep)) } },
            containerColor = colors.surfaceContainerHigh,
        )
    }
}

@Composable
private fun ChatRow(chat: ChatSummary, selected: Boolean, answering: Boolean, onClick: () -> Unit, onDelete: () -> Unit) {
    val strings = LocalContext.current.resources
    val colors = MaterialTheme.colorScheme
    val shape = RoundedCornerShape(16.dp)
    Box(
        Modifier.fillMaxWidth()
            .clip(shape)
            .then(
                if (selected) Modifier
                    .background(Brush.linearGradient(listOf(Teal.copy(alpha = 0.14f), Violet.copy(alpha = 0.10f))))
                    .border(BorderStroke(1.dp, Brush.linearGradient(listOf(Teal.copy(alpha = 0.55f), Violet.copy(alpha = 0.55f)))), shape)
                else Modifier,
            )
            .clickable(onClick = onClick),
    ) {
        Row(Modifier.padding(start = if (selected) 12.dp else 16.dp, top = 8.dp, bottom = 8.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            if (selected) {
                Box(Modifier.width(3.dp).height(28.dp).clip(RoundedCornerShape(50)).background(BrandGradient))
                Spacer(Modifier.width(10.dp))
            }
            Column(Modifier.weight(1f)) {
                Text(chat.title, style = MaterialTheme.typography.bodyLarge, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    color = if (selected) colors.onSecondaryContainer else colors.onSurface)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (answering) {
                        StatusDot(org.equity.app.Phase.Generating)
                        Spacer(Modifier.width(6.dp))
                    }
                    Text(
                        if (answering) strings.getString(R.string.answering)
                        else DateUtils.getRelativeTimeSpanString(chat.updatedAt, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS).toString(),
                        style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant,
                    )
                }
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Outlined.Delete, contentDescription = strings.getString(R.string.delete_chat),
                    tint = if (selected) colors.onSurfaceVariant else colors.outline, modifier = Modifier.size(20.dp))
            }
        }
    }
}
