package org.equity.app.ui

import androidx.compose.ui.platform.LocalContext
import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.equity.app.R
import org.equity.app.Attachment
import org.equity.app.formatBytes
import java.io.File

// Decoded previews, bounded in bytes; attachments are at most 1536 px, so a full-screen view fits easily.
private val previews = object : LruCache<String, ImageBitmap>(24 shl 20) {
    override fun sizeOf(key: String, value: ImageBitmap) = value.width * value.height * 4
}

private fun decode(file: File, maxSide: Int): ImageBitmap? {
    val key = "${file.path}@$maxSide"
    previews.get(key)?.let { return it }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.path, bounds)
    if (bounds.outWidth <= 0) return null
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
    val bitmap = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
    return bitmap.asImageBitmap().also { previews.put(key, it) }
}

@Composable
private fun rememberPreview(file: File, maxSide: Int): ImageBitmap? {
    val image by produceState<ImageBitmap?>(null, file.path, maxSide) {
        value = withContext(Dispatchers.IO) { decode(file, maxSide) }
    }
    return image
}

@Composable
fun ImageThumbnail(file: File, size: Dp, onClick: (() -> Unit)? = null) {
    val pixels = with(LocalDensity.current) { size.roundToPx() }
    val image = rememberPreview(file, pixels * 2)
    val shape = RoundedCornerShape(14.dp)
    Box(
        Modifier.size(size).clip(shape).background(MaterialTheme.colorScheme.surfaceContainerHighest)
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
    ) {
        image?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
    }
}

@Composable
fun DocumentChip(attachment: Attachment, trailing: @Composable (() -> Unit)? = null) {
    val colors = MaterialTheme.colorScheme
    Surface(color = colors.surfaceContainerHighest, shape = RoundedCornerShape(14.dp)) {
        Row(Modifier.padding(start = 10.dp, end = if (trailing == null) 12.dp else 2.dp, top = 8.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Description, contentDescription = null, tint = colors.primary, modifier = Modifier.size(18.dp))
            Spacer(Modifier.size(8.dp))
            Text("${attachment.name} · ${formatBytes(attachment.bytes)}", style = MaterialTheme.typography.labelMedium,
                maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.widthIn(max = 200.dp))
            trailing?.invoke()
        }
    }
}

@Composable
fun ImageViewer(file: File, onDismiss: () -> Unit) {
    val strings = LocalContext.current.resources
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.92f)).clickable(onClick = onDismiss),
            contentAlignment = Alignment.Center) {
            rememberPreview(file, 2048)?.let {
                Image(it, contentDescription = strings.getString(R.string.attached_image), contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize().padding(12.dp))
            }
        }
    }
}
