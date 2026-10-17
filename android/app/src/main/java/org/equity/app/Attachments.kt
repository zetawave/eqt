package org.equity.app

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.Build
import android.os.ext.SdkExtensions
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.util.UUID
import kotlin.math.max
import kotlin.math.roundToInt

class AttachmentException(message: String) : Exception(message)

/** Turns picked files into staged attachments. Call off the UI thread. */
object Attachments {
    // The engine caps an image at 1024 vision tokens (about one megapixel); a longer side of 1536 px keeps
    // detail for the encoder's own resize without storing full-resolution photos.
    private const val MAX_IMAGE_SIDE = 1536
    private const val MAX_PDF_PAGES_AS_IMAGES = 3
    private const val MAX_PDF_TEXT_PAGES = 200
    private const val MAX_TEXT_CHARS = 60_000
    private const val MAX_TEXT_FILE_BYTES = 4L shl 20

    fun import(context: Context, uri: Uri, vision: Boolean): List<Attachment> {
        val resolver = context.contentResolver
        var name = uri.lastPathSegment?.substringAfterLast('/') ?: "file"
        var size = -1L
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                cursor.getString(0)?.let { name = it }
                if (!cursor.isNull(1)) size = cursor.getLong(1)
            }
        }
        val mime = resolver.getType(uri)?.takeIf { it.isNotBlank() } ?: guessType(context, uri, name)
        ChatStore.staging.mkdirs()
        return when {
            mime.startsWith("image/") -> {
                if (!vision) throw AttachmentException("The loaded model cannot read images")
                listOf(image(context, uri, name))
            }
            mime == "application/pdf" || name.endsWith(".pdf", ignoreCase = true) -> pdf(context, uri, name, vision)
            else -> listOf(text(context, uri, name, mime, size))
        }
    }

    /**
     * For sources that report no type (file URIs, some providers): the extension first, then the leading bytes, so a
     * photo is never read as text.
     */
    private fun guessType(context: Context, uri: Uri, name: String): String {
        MimeTypeMap.getSingleton().getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase())?.let { return it }
        val head = ByteArray(12)
        val count = runCatching { context.contentResolver.openInputStream(uri)?.use { it.read(head) } ?: 0 }.getOrDefault(0)
        fun starts(vararg bytes: Int) = count >= bytes.size && bytes.indices.all { head[it] == bytes[it].toByte() }
        fun ascii(at: Int, text: String) = count >= at + text.length && text.indices.all { head[at + it] == text[it].code.toByte() }
        return when {
            starts(0xFF, 0xD8, 0xFF) -> "image/jpeg"
            starts(0x89, 0x50, 0x4E, 0x47) -> "image/png"
            ascii(0, "GIF8") -> "image/gif"
            ascii(0, "RIFF") && ascii(8, "WEBP") -> "image/webp"
            ascii(4, "ftyp") && (ascii(8, "heic") || ascii(8, "heix") || ascii(8, "mif1")) -> "image/heif"
            ascii(0, "%PDF") -> "application/pdf"
            else -> ""
        }
    }

    /** Deletes a staged attachment the user removed before sending. */
    fun discard(attachment: Attachment) {
        File(ChatStore.staging, attachment.file).delete()
    }

    private fun image(context: Context, uri: Uri, name: String): Attachment {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        // ImageDecoder applies EXIF orientation, so the model sees the photo the right way up.
        val bitmap = runCatching {
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val longest = max(info.size.width, info.size.height)
                if (longest > MAX_IMAGE_SIDE) {
                    val scale = MAX_IMAGE_SIDE.toDouble() / longest
                    decoder.setTargetSize((info.size.width * scale).roundToInt().coerceAtLeast(1),
                        (info.size.height * scale).roundToInt().coerceAtLeast(1))
                }
            }
        }.getOrElse { throw AttachmentException("Cannot read this image") }
        return stageJpeg(bitmap, name)
    }

    private fun stageJpeg(bitmap: Bitmap, name: String): Attachment {
        // JPEG has no alpha: transparent pixels are composited on white rather than turning black.
        val opaque = if (bitmap.hasAlpha()) {
            Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888).also {
                Canvas(it).apply { drawColor(Color.WHITE); drawBitmap(bitmap, 0f, 0f, null) }
                bitmap.recycle()
            }
        } else bitmap
        val file = "${UUID.randomUUID()}.jpg"
        val target = File(ChatStore.staging, file)
        target.outputStream().use { opaque.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        opaque.recycle()
        return Attachment(AttachmentKind.Image, name, file, "image/jpeg", target.length())
    }

    @SuppressLint("NewApi")
    private fun pdf(context: Context, uri: Uri, name: String, vision: Boolean): List<Attachment> {
        val descriptor = context.contentResolver.openFileDescriptor(uri, "r")
            ?: throw AttachmentException("Cannot open this PDF")
        return descriptor.use { pfd ->
            val renderer = runCatching { PdfRenderer(pfd) }.getOrElse { throw AttachmentException("Cannot read this PDF") }
            renderer.use {
                // Page text extraction exists from Android 15 (S extension 13).
                val textual = Build.VERSION.SDK_INT >= 35 && SdkExtensions.getExtensionVersion(Build.VERSION_CODES.S) >= 13
                val text = if (!textual) "" else buildString {
                    for (index in 0 until minOf(renderer.pageCount, MAX_PDF_TEXT_PAGES)) {
                        renderer.openPage(index).use { page -> page.textContents.forEach { append(it.text).append('\n') } }
                        if (length > MAX_TEXT_CHARS) break
                    }
                }
                if (text.isNotBlank()) return listOf(stageText(name, "application/pdf", text))
                // Scanned or image-only PDFs: the first pages become images for a vision model.
                if (!vision) throw AttachmentException("This PDF has no readable text, and the loaded model cannot read images")
                (0 until minOf(renderer.pageCount, MAX_PDF_PAGES_AS_IMAGES)).map { index ->
                    renderer.openPage(index).use { page ->
                        val scale = MAX_IMAGE_SIDE.toFloat() / max(page.width, page.height)
                        val bitmap = Bitmap.createBitmap((page.width * scale).roundToInt().coerceAtLeast(1),
                            (page.height * scale).roundToInt().coerceAtLeast(1), Bitmap.Config.ARGB_8888)
                        bitmap.eraseColor(Color.WHITE)
                        page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                        stageJpeg(bitmap, "$name · page ${index + 1}")
                    }
                }
            }
        }
    }

    private fun text(context: Context, uri: Uri, name: String, mime: String, size: Long): Attachment {
        if (size > MAX_TEXT_FILE_BYTES) throw AttachmentException("Text files up to ${formatBytes(MAX_TEXT_FILE_BYTES)} can be attached")
        val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(64 shl 10)
            while (output.size() <= MAX_TEXT_FILE_BYTES) {
                val count = input.read(buffer)
                if (count < 0) break
                output.write(buffer, 0, count)
            }
            output.toByteArray()
        } ?: throw AttachmentException("Cannot open this file")
        if (bytes.size > MAX_TEXT_FILE_BYTES) throw AttachmentException("Text files up to ${formatBytes(MAX_TEXT_FILE_BYTES)} can be attached")
        val text = try {
            Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString()
        } catch (_: CharacterCodingException) {
            null
        }
        if (text == null || text.contains('\u0000')) {
            throw AttachmentException("Only images, PDFs and text files can be attached")
        }
        return stageText(name, mime.ifEmpty { "text/plain" }, text)
    }

    private fun stageText(name: String, mime: String, text: String): Attachment {
        val content = if (text.length <= MAX_TEXT_CHARS) text.trim()
            else text.take(MAX_TEXT_CHARS).trimEnd() + "\n[… truncated: the file continues beyond what fits in the conversation]"
        val file = "${UUID.randomUUID()}.txt"
        val target = File(ChatStore.staging, file)
        target.writeText(content)
        return Attachment(AttachmentKind.Document, name, file, mime, target.length())
    }
}
