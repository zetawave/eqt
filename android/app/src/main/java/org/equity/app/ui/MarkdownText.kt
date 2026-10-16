package org.equity.app.ui

import android.content.ClipData
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.InlineTextContent
import androidx.compose.foundation.text.appendInlineContent
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.Placeholder
import androidx.compose.ui.text.PlaceholderVerticalAlign
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.commonmark.ext.gfm.strikethrough.Strikethrough
import org.commonmark.ext.gfm.strikethrough.StrikethroughExtension
import org.commonmark.ext.gfm.tables.TableBlock
import org.commonmark.ext.gfm.tables.TableCell
import org.commonmark.ext.gfm.tables.TableHead
import org.commonmark.ext.gfm.tables.TableRow
import org.commonmark.ext.gfm.tables.TablesExtension
import org.commonmark.node.BlockQuote
import org.commonmark.node.BulletList
import org.commonmark.node.Code
import org.commonmark.node.Emphasis
import org.commonmark.node.FencedCodeBlock
import org.commonmark.node.HardLineBreak
import org.commonmark.node.Heading
import org.commonmark.node.HtmlBlock
import org.commonmark.node.HtmlInline
import org.commonmark.node.Image as MarkdownImage
import org.commonmark.node.IndentedCodeBlock
import org.commonmark.node.Link
import org.commonmark.node.ListBlock
import org.commonmark.node.ListItem
import org.commonmark.node.Node
import org.commonmark.node.OrderedList
import org.commonmark.node.Paragraph
import org.commonmark.node.SoftLineBreak
import org.commonmark.node.StrongEmphasis
import org.commonmark.node.Text as MarkdownText
import org.commonmark.node.ThematicBreak
import org.commonmark.parser.Parser

private val parser: Parser = Parser.builder()
    .extensions(listOf(TablesExtension.create(), StrikethroughExtension.create()))
    .build()

private val PLACEHOLDER = Regex("[$INLINE_MATH_OPEN$DISPLAY_MATH_OPEN](\\d+)[$INLINE_MATH_CLOSE$DISPLAY_MATH_CLOSE]")

/** A run of inline text, or a display formula that breaks it. */
private sealed interface Segment {
    class Inline(val text: AnnotatedString, val math: List<Int>) : Segment
    class Display(val index: Int) : Segment
}

/** Renders a model answer: CommonMark with GFM tables and strikethrough, plus LaTeX formulas. */
@Composable
fun MarkdownText(markdown: String, color: Color, modifier: Modifier = Modifier) {
    val prepared = remember(markdown) { prepareMarkdown(markdown) }
    val document = remember(prepared) { parser.parse(prepared.markdown) }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Blocks(document, prepared.math, color)
    }
}

@Composable
private fun Blocks(parent: Node, math: List<MathSpan>, color: Color) {
    var node = parent.firstChild
    while (node != null) {
        Block(node, math, color)
        node = node.next
    }
}

@Composable
private fun Block(node: Node, math: List<MathSpan>, color: Color) {
    val typography = MaterialTheme.typography
    when (node) {
        is Paragraph -> Inlines(node, math, typography.bodyLarge, color)
        is Heading -> Inlines(node, math, when (node.level) {
            1 -> typography.titleLarge.copy(fontWeight = FontWeight.SemiBold)
            2 -> typography.titleMedium.copy(fontWeight = FontWeight.SemiBold, fontSize = 19.sp)
            else -> typography.titleMedium.copy(fontWeight = FontWeight.SemiBold)
        }, color)
        is BulletList, is OrderedList -> ListBlockView(node as ListBlock, math, color)
        is FencedCodeBlock -> CodeBlock(node.literal.trimEnd('\n'), node.info?.substringBefore(' ').orEmpty())
        is IndentedCodeBlock -> CodeBlock(node.literal.trimEnd('\n'), "")
        is BlockQuote -> Row(Modifier.height(IntrinsicSize.Min)) {
            Box(Modifier.width(3.dp).fillMaxHeight().background(MaterialTheme.colorScheme.outline, RoundedCornerShape(2.dp)))
            Column(Modifier.padding(start = 12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Blocks(node, math, MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        is ThematicBreak -> HorizontalDivider(Modifier.padding(vertical = 4.dp), color = MaterialTheme.colorScheme.outlineVariant)
        is TableBlock -> TableView(node, math, color)
        is HtmlBlock -> Text(node.literal.trimEnd(), style = typography.bodyLarge, color = color)
        else -> Blocks(node, math, color)
    }
}

@Composable
private fun ListBlockView(list: ListBlock, math: List<MathSpan>, color: Color) {
    val nested = generateSequence(list.parent) { it.parent }.count { it is ListBlock }
    var number = (list as? OrderedList)?.markerStartNumber ?: 1
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        var item = list.firstChild
        while (item != null) {
            if (item is ListItem) {
                val marker = if (list is OrderedList) "${number++}." else if (nested % 2 == 0) "•" else "◦"
                Row {
                    Text(marker, style = MaterialTheme.typography.bodyLarge, color = color,
                        modifier = Modifier.widthIn(min = if (list is OrderedList) 28.dp else 20.dp))
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) { Blocks(item, math, color) }
                }
            }
            item = item.next
        }
    }
}

/** Inline content of a paragraph, heading or table cell; display formulas inside it become centred blocks. */
@Composable
private fun Inlines(parent: Node, math: List<MathSpan>, style: TextStyle, color: Color, modifier: Modifier = Modifier) {
    val colors = MaterialTheme.colorScheme
    val context = LocalContext.current
    val density = LocalDensity.current
    val fontPx = with(density) { style.fontSize.toPx() }
    // Formulas AndroidMath cannot parse are shown as their LaTeX source.
    val segments = remember(parent, style, colors, color) {
        segments(parent, colors.surfaceContainerHighest, colors.primary, math) { index ->
            MathRenderer.render(context, math[index].latex, false, fontPx, color.toArgb()) != null
        }
    }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        for (segment in segments) when (segment) {
            is Segment.Inline -> {
                val inline = HashMap<String, InlineTextContent>()
                for (index in segment.math) {
                    val bitmap = MathRenderer.render(context, math[index].latex, false, fontPx, color.toArgb()) ?: continue
                    val width = with(density) { bitmap.width.toSp() }
                    val height = with(density) { bitmap.height.toSp() }
                    val image = bitmap.asImageBitmap()
                    inline["m$index"] = InlineTextContent(Placeholder(width, height, PlaceholderVerticalAlign.TextCenter)) {
                        Image(image, contentDescription = math[index].latex)
                    }
                }
                Text(segment.text, style = style, color = color, inlineContent = inline)
            }
            is Segment.Display -> DisplayMath(math[segment.index], fontPx * 1.1f, color)
        }
    }
}

@Composable
private fun DisplayMath(span: MathSpan, sizePx: Float, color: Color) {
    val bitmap = MathRenderer.render(LocalContext.current, span.latex, true, sizePx, color.toArgb())
    if (bitmap == null) {
        CodeBlock(span.latex, "latex")
        return
    }
    val density = LocalDensity.current
    Box(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(vertical = 4.dp), contentAlignment = Alignment.Center) {
        Image(bitmap.asImageBitmap(), contentDescription = span.latex,
            modifier = Modifier.size(with(density) { bitmap.width.toDp() }, with(density) { bitmap.height.toDp() }))
    }
}

private fun segments(parent: Node, codeBackground: Color, linkColor: Color, math: List<MathSpan>,
                     renderable: (Int) -> Boolean): List<Segment> {
    val result = ArrayList<Segment>()
    var builder = AnnotatedString.Builder()
    var inlineMath = ArrayList<Int>()
    fun flush() {
        if (builder.length > 0) result += Segment.Inline(builder.toAnnotatedString(), inlineMath)
        builder = AnnotatedString.Builder()
        inlineMath = ArrayList()
    }
    fun literal(text: String) {
        var last = 0
        for (match in PLACEHOLDER.findAll(text)) {
            builder.append(text.substring(last, match.range.first))
            val index = match.groupValues[1].toInt()
            if (match.value[0] == DISPLAY_MATH_OPEN) {
                flush()
                result += Segment.Display(index)
            } else if (!renderable(index)) {
                builder.withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = codeBackground, fontSize = 0.9.em)) {
                    append(math[index].latex)
                }
            } else {
                builder.appendInlineContent("m$index", "￼")
                inlineMath += index
            }
            last = match.range.last + 1
        }
        builder.append(text.substring(last))
    }
    fun walk(node: Node) {
        var child = node.firstChild
        while (child != null) {
            when (child) {
                is MarkdownText -> literal(child.literal)
                is SoftLineBreak -> builder.append(' ')
                is HardLineBreak -> builder.append('\n')
                is Code -> builder.withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = codeBackground, fontSize = 0.9.em)) {
                    append(" ${child.literal} ")
                }
                is Emphasis -> builder.withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { walk(child) }
                is StrongEmphasis -> builder.withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { walk(child) }
                is Strikethrough -> builder.withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { walk(child) }
                is Link -> builder.withLink(LinkAnnotation.Url(child.destination,
                    TextLinkStyles(SpanStyle(color = linkColor, textDecoration = TextDecoration.Underline)))) { walk(child) }
                is MarkdownImage -> walk(child)
                is HtmlInline -> builder.append(child.literal)
                else -> walk(child)
            }
            child = child.next
        }
    }
    walk(parent)
    flush()
    return result
}

@Composable
private fun CodeBlock(code: String, language: String) {
    val colors = MaterialTheme.colorScheme
    val clipboard = LocalClipboard.current
    val scope = rememberCoroutineScope()
    Surface(color = colors.surfaceContainerHighest, shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
        Column {
            Row(Modifier.fillMaxWidth().padding(start = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(language.ifEmpty { "code" }, style = MaterialTheme.typography.labelMedium, color = colors.onSurfaceVariant,
                    modifier = Modifier.weight(1f))
                IconButton(onClick = { scope.launch { clipboard.setClipEntry(ClipEntry(ClipData.newPlainText("Code", code))) } },
                    modifier = Modifier.size(36.dp)) {
                    Icon(Icons.Outlined.ContentCopy, contentDescription = "Copy code", tint = colors.onSurfaceVariant, modifier = Modifier.size(16.dp))
                }
            }
            Text(code, style = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace, fontSize = 13.sp, lineHeight = 19.sp),
                color = colors.onSurface, softWrap = false,
                modifier = Modifier.horizontalScroll(rememberScrollState()).padding(start = 14.dp, end = 14.dp, bottom = 12.dp))
        }
    }
}

@Composable
private fun TableView(table: TableBlock, math: List<MathSpan>, color: Color) {
    val colors = MaterialTheme.colorScheme
    val style = MaterialTheme.typography.bodyMedium
    val rows = remember(table) {
        val list = ArrayList<Pair<Boolean, List<TableCell>>>()
        fun collect(node: Node, header: Boolean) {
            var child = node.firstChild
            while (child != null) {
                if (child is TableRow) list += header to generateSequence(child.firstChild) { it.next }.filterIsInstance<TableCell>().toList()
                else collect(child, child is TableHead)
                child = child.next
            }
        }
        collect(table, false)
        list
    }
    val columns = rows.maxOfOrNull { it.second.size } ?: 0
    // Columns are as wide as their widest cell, up to a cap after which cells wrap; the table scrolls sideways.
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    val widths = remember(table, style) {
        (0 until columns).map { column ->
            val widest = rows.maxOf { (header, cells) ->
                cells.getOrNull(column)?.let { cell ->
                    val text = segments(cell, colors.surfaceContainerHighest, colors.primary, math) { true }.filterIsInstance<Segment.Inline>()
                        .joinToString(" ") { it.text.text }
                    measurer.measure(text, style.copy(fontWeight = if (header) FontWeight.SemiBold else null)).size.width
                } ?: 0
            }
            with(density) { (widest.toDp() + 25.dp).coerceIn(48.dp, 260.dp) }
        }
    }
    Box(Modifier.horizontalScroll(rememberScrollState())) {
        Column(Modifier.border(1.dp, colors.outlineVariant, RoundedCornerShape(10.dp))) {
            rows.forEachIndexed { r, (header, cells) ->
                if (r > 0) HorizontalDivider(color = colors.outlineVariant)
                Row(Modifier.height(IntrinsicSize.Min).then(if (header) Modifier.background(colors.surfaceContainerHigh) else Modifier)) {
                    for (column in 0 until columns) {
                        Box(Modifier.width(widths[column]).fillMaxHeight().padding(horizontal = 12.dp, vertical = 8.dp)) {
                            cells.getOrNull(column)?.let {
                                Inlines(it, math, if (header) style.copy(fontWeight = FontWeight.SemiBold) else style, color)
                            }
                        }
                    }
                }
            }
        }
    }
}
