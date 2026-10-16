package org.equity.app.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.util.LruCache
import android.view.View
import com.agog.mathdisplay.MTFontManager
import com.agog.mathdisplay.MTMathView
import com.agog.mathdisplay.parse.MTParseErrors

/** A formula taken out of an answer before Markdown parsing; [display] formulas get their own centred line. */
class MathSpan(val latex: String, val display: Boolean)

/** Markdown with every formula replaced by a private-use placeholder, and the formulas by index. */
class PreparedMarkdown(val markdown: String, val math: List<MathSpan>)

// Placeholders use Private Use Area characters, which Markdown treats as plain text and models do not emit.
const val INLINE_MATH_OPEN = ''
const val INLINE_MATH_CLOSE = ''
const val DISPLAY_MATH_OPEN = ''
const val DISPLAY_MATH_CLOSE = ''

private val FENCE = Regex("^(`{3,}|~{3,})")

/**
 * Finds `$$…$$`, `\[…\]`, `\(…\)` and `$…$` outside code. Markdown would otherwise eat LaTeX backslashes,
 * underscores and asterisks. A single `$` opens a formula only when followed by a non-space and closed on the same
 * line by a `$` that follows a non-space and is not followed by a digit, so prices such as "$5 and $10" stay text.
 * An unclosed formula (an answer still streaming) stays text until its delimiter arrives.
 */
fun prepareMarkdown(text: String): PreparedMarkdown {
    val out = StringBuilder(text.length)
    val math = mutableListOf<MathSpan>()
    val prose = StringBuilder()
    var fence: String? = null
    val lines = text.split('\n')
    for ((index, line) in lines.withIndex()) {
        val newline = if (index < lines.lastIndex) "\n" else ""
        val trimmed = line.trimStart()
        val indented = line.length - trimmed.length >= 4
        if (fence == null) {
            val marker = FENCE.find(trimmed)?.value
            if (marker != null && !indented) {
                scanProse(prose, out, math)
                fence = marker
                out.append(line).append(newline)
            } else {
                prose.append(line).append(newline)
            }
        } else {
            out.append(line).append(newline)
            val body = trimmed.trimEnd()
            if (!indented && body.length >= fence.length && body.all { it == fence!![0] }) fence = null
        }
    }
    scanProse(prose, out, math)
    return PreparedMarkdown(out.toString(), math)
}

private fun scanProse(prose: StringBuilder, out: StringBuilder, math: MutableList<MathSpan>) {
    val s = prose
    var i = 0
    fun emit(latex: String, display: Boolean) {
        out.append(if (display) DISPLAY_MATH_OPEN else INLINE_MATH_OPEN).append(math.size)
            .append(if (display) DISPLAY_MATH_CLOSE else INLINE_MATH_CLOSE)
        math += MathSpan(latex.trim(), display)
    }
    while (i < s.length) {
        val c = s[i]
        when {
            c == '`' -> {
                // Inline code is copied verbatim up to the matching backtick run.
                var run = 0
                while (i + run < s.length && s[i + run] == '`') run++
                val closing = s.indexOf("`".repeat(run), i + run)
                val end = if (closing < 0) i + run else closing + run
                out.append(s, i, end)
                i = end
            }
            c == '\\' && i + 1 < s.length && (s[i + 1] == '[' || s[i + 1] == '(') -> {
                val close = if (s[i + 1] == '[') "\\]" else "\\)"
                val end = s.indexOf(close, i + 2)
                if (end < 0) {
                    out.append(s, i, i + 2)
                    i += 2
                } else {
                    emit(s.substring(i + 2, end), s[i + 1] == '[')
                    i = end + 2
                }
            }
            c == '\\' && i + 1 < s.length -> {
                out.append(s, i, i + 2)
                i += 2
            }
            c == '$' && i + 1 < s.length && s[i + 1] == '$' -> {
                val end = s.indexOf("$$", i + 2)
                if (end < 0 || s.substring(i + 2, end).isBlank()) {
                    out.append("$$")
                    i += 2
                } else {
                    emit(s.substring(i + 2, end), true)
                    i = end + 2
                }
            }
            c == '$' -> {
                val end = inlineDollarEnd(s, i)
                if (end < 0) {
                    out.append(c)
                    i++
                } else {
                    emit(s.substring(i + 1, end), false)
                    i = end + 1
                }
            }
            else -> {
                out.append(c)
                i++
            }
        }
    }
    prose.clear()
}

private fun inlineDollarEnd(s: CharSequence, start: Int): Int {
    if (start + 1 >= s.length || s[start + 1].isWhitespace()) return -1
    var j = start + 1
    while (j < s.length && s[j] != '\n') {
        if (s[j] == '\\') {
            j += 2
            continue
        }
        if (s[j] == '$') {
            val closes = !s[j - 1].isWhitespace() && (j + 1 >= s.length || !s[j + 1].isDigit())
            return if (closes && j > start + 1) j else -1
        }
        j++
    }
    return -1
}

/** Draws formulas with AndroidMath into bitmaps that Compose places inline or as blocks. Main thread only. */
object MathRenderer {
    // Bitmaps are small (one formula each); bounded in bytes.
    private val cache = object : LruCache<String, Bitmap>(16 shl 20) {
        override fun sizeOf(key: String, value: Bitmap) = value.byteCount
    }
    private val failed = HashSet<String>()
    private var fontsReady = false

    /** Null when AndroidMath cannot parse the formula; the caller then shows its source. */
    fun render(context: Context, latex: String, display: Boolean, sizePx: Float, color: Int): Bitmap? {
        val key = "$display|$sizePx|$color|$latex"
        cache.get(key)?.let { return it }
        if (key in failed) return null
        if (!fontsReady) {
            MTFontManager.setContext(context.applicationContext)
            fontsReady = true
        }
        val bitmap = runCatching {
            val view = MTMathView(context).apply {
                displayErrorInline = false
                fontSize = sizePx
                textColor = color
                labelMode = if (display) MTMathView.MTMathViewMode.KMTMathViewModeDisplay
                    else MTMathView.MTMathViewMode.KMTMathViewModeText
                this.latex = latex
            }
            if (view.lastError.errorcode != MTParseErrors.ErrorNone) return@runCatching null
            val unbounded = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
            view.measure(unbounded, unbounded)
            if (view.measuredWidth <= 0 || view.measuredHeight <= 0) return@runCatching null
            view.layout(0, 0, view.measuredWidth, view.measuredHeight)
            Bitmap.createBitmap(view.measuredWidth, view.measuredHeight, Bitmap.Config.ARGB_8888).also { view.draw(Canvas(it)) }
        }.getOrNull()
        if (bitmap == null) failed += key else cache.put(key, bitmap)
        return bitmap
    }
}
