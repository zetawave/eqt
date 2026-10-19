package org.equity.app

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The byte layout of llama.cpp's gguf-split merge, built from the shards' headers (tools/gguf_merge.py is the
 * reference and checks it against a real merge). The merged file is the first shard's metadata with
 * `split.count` moved last and set to 0, every shard's tensor infos in shard order with offsets recomputed at
 * 32-byte alignment, zero padding to 32, then each shard's data section in shard order. A download can
 * therefore write the merged header first and append the shards' data sections as they arrive.
 */
object GgufMerge {
    private const val ALIGNMENT = 32L
    private val SCALAR_BYTES = mapOf(0 to 1, 1 to 1, 2 to 2, 3 to 2, 4 to 4, 5 to 4, 6 to 4, 7 to 1, 10 to 8, 11 to 8, 12 to 8)

    class Shard(
        val dataOffset: Long,
        internal val entries: List<Pair<String, ByteArray>>,
        /** Each tensor's info without its offset, and the padded bytes it occupies in the data section. */
        internal val tensors: List<Pair<ByteArray, Long>>,
    ) {
        val dataBytes get() = tensors.sumOf { it.second }
    }

    class Truncated : Exception()

    /** Parses a shard header; [shardSize] gives the last tensor's extent. Throws [Truncated] if [bytes] ends early. */
    fun parse(bytes: ByteArray, shardSize: Long): Shard {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        fun need(count: Long) {
            if (count < 0 || buffer.remaining() < count) throw Truncated()
        }
        fun u32(): Long { need(4); return buffer.int.toLong() and 0xffffffffL }
        fun u64(): Long { need(8); return buffer.long.also { require(it >= 0) { "GGUF count out of range" } } }
        fun skip(count: Long) { need(count); buffer.position(buffer.position() + count.toInt()) }
        fun string(): String {
            val length = u64()
            need(length)
            val text = String(bytes, buffer.position(), length.toInt(), Charsets.UTF_8)
            skip(length)
            return text
        }
        fun skipValue(kind: Int, nested: Boolean) {
            val scalar = SCALAR_BYTES[kind]
            when {
                scalar != null -> skip(scalar.toLong())
                kind == 8 -> skip(u64())
                kind == 9 && !nested -> {
                    val element = u32().toInt()
                    val count = u64()
                    val elementBytes = SCALAR_BYTES[element]
                    if (elementBytes != null) skip(count * elementBytes) else repeat(count.toInt()) { skipValue(element, true) }
                }
                else -> throw IllegalArgumentException("Unsupported GGUF metadata type $kind")
            }
        }
        need(24)
        require(buffer.int == 0x46554747 && buffer.int == 3) { "Expected a GGUF v3 shard" }
        val tensorCount = buffer.long
        val entryCount = buffer.long
        require(tensorCount in 0..1_000_000 && entryCount in 0..100_000) { "Implausible GGUF header" }
        val entries = ArrayList<Pair<String, ByteArray>>(entryCount.toInt())
        repeat(entryCount.toInt()) {
            val start = buffer.position()
            val key = string()
            val kind = u32().toInt()
            if (key == "general.alignment") {
                need(4)
                require(buffer.getInt(buffer.position()).toLong() == ALIGNMENT) { "Only 32-byte GGUF alignment is merged" }
            }
            skipValue(kind, false)
            entries += key to bytes.copyOfRange(start, buffer.position())
        }
        val infos = ArrayList<ByteArray>(tensorCount.toInt())
        val offsets = ArrayList<Long>(tensorCount.toInt())
        repeat(tensorCount.toInt()) {
            val start = buffer.position()
            string()
            val dims = u32()
            require(dims in 1..4) { "Invalid tensor rank" }
            skip(8 * dims + 4)
            infos += bytes.copyOfRange(start, buffer.position())
            offsets += u64()
        }
        val dataOffset = (buffer.position() + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT
        val dataBytes = shardSize - dataOffset
        require(dataBytes >= 0) { "Shard is smaller than its header" }
        // Tensors are stored in info order, each padded to the alignment, as gguf_merge writes them.
        val tensors = infos.indices.map { i ->
            val end = if (i + 1 < offsets.size) offsets[i + 1] else dataBytes
            require(offsets[i] % ALIGNMENT == 0L && end > offsets[i] && (i > 0 || offsets[0] == 0L)) { "Unexpected shard layout" }
            infos[i] to end - offsets[i]
        }
        if (tensors.isEmpty()) require(dataBytes == 0L) { "Shard has data but no tensors" }
        return Shard(dataOffset, entries, tensors)
    }

    fun mergedHeader(shards: List<Shard>): ByteArray {
        val first = shards.first().entries
        require(first.any { it.first == "split.count" }) { "The first shard has no split.count" }
        val entries = first.filter { it.first != "split.count" }.map { it.second } + splitCountZero()
        val tensors = shards.flatMap { it.tensors }
        var size = 24L + entries.sumOf { it.size.toLong() } + tensors.sumOf { it.first.size + 8L }
        size = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT
        require(size <= Int.MAX_VALUE) { "GGUF header too large" }
        val out = ByteBuffer.allocate(size.toInt()).order(ByteOrder.LITTLE_ENDIAN)
        out.putInt(0x46554747).putInt(3).putLong(tensors.size.toLong()).putLong(entries.size.toLong())
        entries.forEach { out.put(it) }
        var offset = 0L
        for ((info, bytes) in tensors) {
            out.put(info).putLong(offset)
            offset += bytes
        }
        return out.array()
    }

    private fun splitCountZero(): ByteArray {
        val key = "split.count".toByteArray()
        return ByteBuffer.allocate(8 + key.size + 4 + 2).order(ByteOrder.LITTLE_ENDIAN)
            .putLong(key.size.toLong()).put(key).putInt(2).putShort(0).array()
    }
}
