package org.equity.app.ui

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.IconButton
import androidx.compose.material3.Icon
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import org.equity.app.R

// The ambient-premium building blocks shared by every screen: a flat background, a translucent card,
// gradient-filled and bordered buttons, gradient hero text and a thin gradient progress track.

@Composable
private fun HelpBadge(modifier: Modifier = Modifier) {
    Box(
        modifier.size(18.dp).border(1.dp, MaterialTheme.colorScheme.outline, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Text("?", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
fun HelpIconButton(title: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val description = stringResource(R.string.help_for, title)
    IconButton(onClick = onClick, modifier = modifier.semantics { contentDescription = description }) {
        HelpBadge()
    }
}

@Composable
fun HelpButton(title: String, text: String, modifier: Modifier = Modifier) {
    var open by rememberSaveable { mutableStateOf(false) }
    HelpIconButton(title, { open = true }, modifier)
    if (open) HelpDialog(title, { open = false }) {
        Text(text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
fun HelpLabel(
    title: String,
    text: String,
    summary: String,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.labelLarge,
) {
    Column(modifier) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(title, style = style, modifier = Modifier.weight(1f, fill = false))
            HelpButton(title, text)
        }
        Text(summary, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Bounds the scroll area to the dialog window so large fonts and long explanations keep Close reachable. */
@Composable
fun HelpDialog(title: String, onDismiss: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        BoxWithConstraints(Modifier.fillMaxWidth().padding(24.dp), contentAlignment = Alignment.Center) {
            PremiumCard(Modifier.width(440.dp).heightIn(max = maxHeight), tint = true, opaque = true) {
                Box(Modifier.width(40.dp).height(3.dp).clip(CircleShape).background(BrandGradient))
                Spacer(Modifier.height(16.dp))
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(title, style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.weight(1f).semantics { heading() })
                }
                Spacer(Modifier.height(16.dp))
                Column(
                    Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    content = content,
                )
                Spacer(Modifier.height(20.dp))
                GradientButton(onClick = onDismiss, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                    Text(stringResource(R.string.close), style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }
}

/** Flat background fill; kept as a shared modifier so every screen roots on the same base. */
@Composable
fun Modifier.screenBackground(): Modifier {
    return this.background(MaterialTheme.colorScheme.background)
}

/** A translucent rounded surface with a hairline border; [tint] adds a faint gradient wash from the top edge. */
@Composable
fun PremiumCard(
    modifier: Modifier = Modifier,
    tint: Boolean = false,
    opaque: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) {
    val container = MaterialTheme.colorScheme.surfaceContainerLow.copy(alpha = if (opaque) 1f else 0.78f)
    val border = MaterialTheme.colorScheme.outlineVariant
    val shape = RoundedCornerShape(26.dp)
    Box(
        Modifier
            .then(modifier)
            .clip(shape)
            .background(container)
            .drawBehind {
                if (tint) {
                    drawRoundRect(
                        brush = Brush.verticalGradient(
                            listOf(Teal.copy(alpha = 0.07f), Color.Transparent),
                            startY = 0f, endY = size.height * 0.6f,
                        ),
                        cornerRadius = CornerRadius(26.dp.toPx()),
                    )
                }
                drawRoundRect(color = border, cornerRadius = CornerRadius(26.dp.toPx()), style = Stroke(1.dp.toPx()))
            },
    ) {
        Column(Modifier.fillMaxWidth().padding(20.dp), content = content)
    }
}

/** Hero copy set in the brand gradient. For large display text only, never body copy. */
@Composable
fun GradientText(
    text: String,
    style: TextStyle,
    modifier: Modifier = Modifier,
    textAlign: TextAlign? = null,
) {
    Text(text, style = style.copy(brush = BrandGradient), modifier = modifier, textAlign = textAlign)
}

/** The primary action: filled with the brand gradient, dark label, discreet 40dp height, optional leading icon. */
@Composable
fun GradientButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null,
    compact: Boolean = false,
    content: @Composable RowScope.() -> Unit,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val fillAlpha by animateFloatAsState(if (enabled) if (pressed) 0.85f else 1f else 0.38f, label = "gradientButtonAlpha")
    val shape = RoundedCornerShape(if (compact) 12.dp else 16.dp)
    val onPrimary = MaterialTheme.colorScheme.onPrimary
    Box(
        modifier
            .minimumInteractiveComponentSize()
            .heightIn(min = if (compact) 36.dp else 40.dp)
            .clip(shape)
            .alpha(fillAlpha)
            .background(BrandGradient)
            .clickable(interactionSource = interaction, indication = ripple(), enabled = enabled, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Row(
            Modifier.padding(horizontal = 16.dp, vertical = if (compact) 6.dp else 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CompositionLocalProvider(LocalContentColor provides onPrimary) {
                if (icon != null) {
                    Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                }
                content()
            }
        }
    }
}

/** The secondary action: translucent with a hairline border, same geometry as [GradientButton]. */
@Composable
fun PremiumOutlinedButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    contentColor: Color = MaterialTheme.colorScheme.onSurface,
    compact: Boolean = false,
    content: @Composable RowScope.() -> Unit,
) {
    val shape = RoundedCornerShape(if (compact) 12.dp else 16.dp)
    val colors = MaterialTheme.colorScheme
    Box(
        modifier
            .minimumInteractiveComponentSize()
            .heightIn(min = if (compact) 36.dp else 48.dp)
            .clip(shape)
            .alpha(if (enabled) 1f else 0.5f)
            .background(colors.surfaceContainerLow.copy(alpha = 0.6f))
            .border(1.dp, colors.outline, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Row(
            Modifier.padding(horizontal = if (compact) 16.dp else 20.dp, vertical = if (compact) 6.dp else 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CompositionLocalProvider(LocalContentColor provides contentColor) { content() }
        }
    }
}

/** Section label: small uppercase in muted color with wide tracking. */
@Composable
fun SectionHeader(text: String, modifier: Modifier = Modifier) {
    Text(
        text.uppercase(),
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        letterSpacing = TextUnit(0.12f, TextUnitType.Em),
        modifier = modifier,
    )
}

/** A thin rounded track whose fill sweeps the brand gradient; [progress] null slides a segment across the track. */
@Composable
fun GradientProgress(progress: Float?, modifier: Modifier = Modifier) {
    val track = MaterialTheme.colorScheme.surfaceContainerHighest
    val target = (progress ?: 0f).coerceIn(0f, 1f)
    val animated by animateFloatAsState(target, animationSpec = tween(240, easing = FastOutSlowInEasing), label = "gradientProgress")
    Box(modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(50)).background(track)) {
        if (progress != null) {
            Box(
                Modifier
                    .fillMaxWidth(animated)
                    .height(6.dp)
                    .clip(RoundedCornerShape(50))
                    .background(BrandGradient),
            )
        } else {
            val transition = rememberInfiniteTransition(label = "indeterminateProgress")
            val shift by transition.animateFloat(
                initialValue = -0.35f, targetValue = 1.05f,
                animationSpec = infiniteRepeatable(tween(1100, easing = FastOutSlowInEasing), repeatMode = RepeatMode.Restart),
                label = "shift",
            )
            BoxWithConstraints(Modifier.fillMaxWidth().height(6.dp)) {
                val segment = maxWidth * 0.35f
                Box(
                    Modifier
                        .width(segment)
                        .height(6.dp)
                        .offset(x = (maxWidth + segment) * shift)
                        .clip(RoundedCornerShape(50))
                        .background(BrandGradient),
                )
            }
        }
    }
}

/** A circular gauge with a gradient sweep, for the context-window share; [warn] switches it to the error color. */
@Composable
fun GradientGauge(share: Float, modifier: Modifier = Modifier, size: Dp = 20.dp, stroke: Dp = 2.5.dp, warn: Boolean = false) {
    val colors = MaterialTheme.colorScheme
    Box(modifier.size(28.dp), contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(size)) {
            drawArc(colors.surfaceContainerHighest, startAngle = 0f, sweepAngle = 360f, useCenter = false,
                style = Stroke(stroke.toPx()))
            if (warn) {
                drawArc(colors.error, startAngle = -90f, sweepAngle = 360f * share.coerceIn(0f, 1f),
                    useCenter = false, style = Stroke(stroke.toPx(), cap = StrokeCap.Round))
            } else {
                drawArc(Brush.sweepGradient(listOf(Teal, Violet)), startAngle = -90f,
                    sweepAngle = 360f * share.coerceIn(0f, 1f), useCenter = false,
                    style = Stroke(stroke.toPx(), cap = StrokeCap.Round))
            }
        }
    }
}

