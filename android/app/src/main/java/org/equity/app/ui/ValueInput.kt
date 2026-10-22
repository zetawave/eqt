package org.equity.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.roundToInt

/** [value] with at most [decimals] decimals and no trailing zeros ("0.7", "0.65", "1"). */
fun formatDecimal(value: Float, decimals: Int): String =
    "%.${decimals}f".format(Locale.US, value).trimEnd('0').trimEnd('.').let { if (it == "-0") "0" else it }

/**
 * A number set with a slider over [range] in steps of [step], or typed exactly in the field beside it. A typed value
 * anywhere in the range is kept as typed; one outside it, or one [accept] refuses, is marked and not applied.
 */
@Composable
fun SliderField(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float>,
    step: Float,
    format: (Float) -> String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    integer: Boolean = false,
    accept: (Float) -> Boolean = { true },
    hint: String? = null,
) {
    val strings = LocalContext.current.resources
    var text by remember { mutableStateOf(format(value)) }
    val typed = text.trim().replace(',', '.').toFloatOrNull()
    val valid = typed != null && typed in range && accept(typed) && (!integer || typed == typed.roundToInt().toFloat())
    // The slider and the caller can change the value; the field follows unless it already shows that value.
    LaunchedEffect(value) { if (typed != value) text = format(value) }
    Column(modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Slider(
                value = value.coerceIn(range.start, range.endInclusive),
                onValueChange = { raw ->
                    val snapped = range.start + ((raw - range.start) / step).roundToInt() * step
                    // Through the displayed text, so a step of 0.05 stores 0.05 and not 0.050000012.
                    val next = format(snapped.coerceIn(range.start, range.endInclusive)).toFloatOrNull()
                        ?: snapped.coerceIn(range.start, range.endInclusive)
                    text = format(next)
                    onValueChange(next)
                },
                valueRange = range,
                steps = (((range.endInclusive - range.start) / step).roundToInt() - 1).coerceAtLeast(0),
                enabled = enabled,
                modifier = Modifier.weight(1f),
            )
            OutlinedTextField(
                value = text,
                onValueChange = { next ->
                    text = next
                    val parsed = next.trim().replace(',', '.').toFloatOrNull()
                    if (parsed != null && parsed in range && accept(parsed) &&
                        (!integer || parsed == parsed.roundToInt().toFloat())) onValueChange(parsed)
                },
                singleLine = true,
                enabled = enabled,
                isError = !valid,
                keyboardOptions = KeyboardOptions(
                    keyboardType = if (integer) KeyboardType.Number else KeyboardType.Decimal, imeAction = ImeAction.Done),
                textStyle = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.width(96.dp),
            )
        }
        hint?.let {
            Text(it, style = MaterialTheme.typography.bodySmall,
                color = if (valid) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error)
        }
    }
}
