@file:OptIn(ExperimentalTextApi::class)

package org.equity.app.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.equity.app.R

private fun inter(weight: Int) = Font(
    R.font.inter,
    weight = FontWeight(weight),
    variationSettings = FontVariation.Settings(FontVariation.weight(weight)),
)

val Inter = FontFamily(inter(400), inter(500), inter(600), inter(700))

val Teal = Color(0xFF2DD4BF)
val Violet = Color(0xFF8B5CF6)
val BrandGradient = Brush.linearGradient(listOf(Teal, Violet))

private val Colors = darkColorScheme(
    primary = Color(0xFF5EEAD4),
    onPrimary = Color(0xFF032B26),
    primaryContainer = Color(0xFF123F3B),
    onPrimaryContainer = Color(0xFFCCFBF1),
    secondary = Color(0xFFB4A2FB),
    onSecondary = Color(0xFF1F1240),
    secondaryContainer = Color(0xFF2A2148),
    onSecondaryContainer = Color(0xFFEDE9FE),
    tertiary = Color(0xFFFBBF77),
    background = Color(0xFF0E1014),
    onBackground = Color(0xFFE7E9EE),
    surface = Color(0xFF0E1014),
    onSurface = Color(0xFFE7E9EE),
    surfaceVariant = Color(0xFF1C2029),
    onSurfaceVariant = Color(0xFF9CA4B4),
    surfaceContainerLowest = Color(0xFF0A0C0F),
    surfaceContainerLow = Color(0xFF121519),
    surfaceContainer = Color(0xFF161A20),
    surfaceContainerHigh = Color(0xFF1C2028),
    surfaceContainerHighest = Color(0xFF242934),
    outline = Color(0xFF3A4150),
    outlineVariant = Color(0xFF262B35),
    error = Color(0xFFF87171),
    onError = Color(0xFF3B0A0A),
    errorContainer = Color(0xFF4A1D1F),
    onErrorContainer = Color(0xFFFECACA),
)

private fun TextStyle.interStyle(size: TextUnit? = null, weight: Int? = null, height: TextUnit? = null) = copy(
    fontFamily = Inter,
    fontSize = size ?: fontSize,
    fontWeight = weight?.let { FontWeight(it) } ?: fontWeight,
    lineHeight = height ?: lineHeight,
)

private val Base = Typography()
private val Type = Typography(
    displayLarge = Base.displayLarge.interStyle(weight = 600),
    displayMedium = Base.displayMedium.interStyle(weight = 600),
    displaySmall = Base.displaySmall.interStyle(weight = 600),
    headlineLarge = Base.headlineLarge.interStyle(weight = 600),
    headlineMedium = Base.headlineMedium.interStyle(size = 26.sp, weight = 600),
    headlineSmall = Base.headlineSmall.interStyle(size = 22.sp, weight = 600),
    titleLarge = Base.titleLarge.interStyle(size = 20.sp, weight = 600),
    titleMedium = Base.titleMedium.interStyle(weight = 600),
    titleSmall = Base.titleSmall.interStyle(weight = 600),
    bodyLarge = Base.bodyLarge.interStyle(size = 16.sp, height = 24.sp),
    bodyMedium = Base.bodyMedium.interStyle(size = 14.sp, height = 21.sp),
    bodySmall = Base.bodySmall.interStyle(size = 12.sp, height = 17.sp),
    labelLarge = Base.labelLarge.interStyle(weight = 600),
    labelMedium = Base.labelMedium.interStyle(weight = 500),
    labelSmall = Base.labelSmall.interStyle(weight = 500),
)

@Composable
fun EquityTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Colors, typography = Type, content = content)
}

/** The Equity mark (equals sign on the brand gradient) with an optional wordmark. */
@Composable
fun Logo(size: Dp, wordmark: Boolean = true, modifier: Modifier = Modifier) {
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Image(painterResource(R.drawable.ic_equity_mark), contentDescription = "Equity", modifier = Modifier.size(size))
        if (wordmark) {
            Spacer(Modifier.width(size * 0.32f))
            Text(
                "Equity",
                style = TextStyle(fontFamily = Inter, fontWeight = FontWeight(650), fontSize = (size.value * 0.62f).sp,
                    letterSpacing = (-0.02f * size.value).sp, brush = BrandGradient),
            )
        }
    }
}
