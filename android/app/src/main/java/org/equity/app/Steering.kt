package org.equity.app

import android.content.Context
import android.net.Uri
import android.util.AtomicFile
import org.json.JSONObject
import java.io.File
import java.util.UUID

data class SteeringOptions(val enabled: Boolean = false, val strength: Float = 1f) {
    companion object {
        /** Negative values project the direction out of every block (-1 removes it); positive values add it. */
        val range = -1f..2f
        fun valid(strength: Float) = strength.isFinite() && strength != 0f && strength in range
    }
}

data class SteeringProfile(val metadata: String) {
    private val json get() = JSONObject(metadata)
    val modelHash get() = json.getString("model_sha256")
    val vectorHash get() = json.getString("vector_sha256")
    val layer get() = json.getInt("layer_index")
    val embedding get() = json.getInt("embedding_size")
    val architecture get() = json.getString("architecture")
    val direction get() = json.getString("direction")
    fun vector(context: Context) = File(context.filesDir, "steering/$modelHash/$vectorHash/vector.gguf")
    fun request(context: Context, strength: Float): JSONObject = JSONObject()
        .put("path", vector(context).absolutePath).put("metadata", json).put("strength", strength.toDouble())
}

/** What the Models screen shows for a model's imported vector; the vector is verified natively when the model loads. */
data class SteeringSummary(val vectorHash: String, val layer: Int, val direction: String)

/**
 * Imports, reads and removals run on EngineController's worker. Imported files are copied, verified, then published
 * atomically, so [summary] can read the published JSON from any thread for display.
 */
object SteeringFiles {
    private const val VECTOR_LIMIT = 8 * 1024 * 1024
    private const val METADATA_LIMIT = 1024 * 1024
    private val hashPattern = Regex("[0-9a-f]{64}")

    private fun root(context: Context) = File(context.filesDir, "steering").apply { mkdirs() }
    private fun modelDirectory(context: Context, hash: String): File {
        require(hashPattern.matches(hash)) { context.getString(R.string.invalid_model_fingerprint) }
        return File(root(context), hash)
    }

    fun read(context: Context, modelHash: String): SteeringProfile? {
        val folder = modelDirectory(context, modelHash)
        val index = File(folder, "current.json")
        if (!index.exists() && !File(index.path + ".bak").exists()) return null
        require(index.length() <= 4096) { context.getString(R.string.invalid_steering_index) }
        val vectorHash = JSONObject(AtomicFile(index).openRead().bufferedReader().use { it.readText() }).getString("vector_sha256")
        require(hashPattern.matches(vectorHash)) { context.getString(R.string.invalid_vector_fingerprint) }
        val directory = File(folder, vectorHash)
        val manifest = File(directory, "metadata.json")
        require(manifest.length() in 1..METADATA_LIMIT.toLong()) { context.getString(R.string.missing_or_invalid_steering_metadata) }
        val info = NativeBridge.inspectSteering(File(directory, "vector.gguf").absolutePath.toByteArray(), manifest.readBytes())
        val profile = SteeringProfile(info.decodeToString())
        require(profile.modelHash == modelHash && profile.vectorHash == vectorHash) { context.getString(R.string.steering_association_mismatch) }
        return profile
    }

    /** The imported vector associated with [modelHash], from its published metadata; null when there is none. */
    fun summary(context: Context, modelHash: String): SteeringSummary? = runCatching {
        val folder = modelDirectory(context, modelHash)
        val index = File(folder, "current.json")
        if (!index.exists() && !File(index.path + ".bak").exists()) return@runCatching null
        val vectorHash = JSONObject(AtomicFile(index).openRead().bufferedReader().use { it.readText() }).getString("vector_sha256")
        require(hashPattern.matches(vectorHash))
        val manifest = File(File(folder, vectorHash), "metadata.json")
        require(manifest.length() in 1..METADATA_LIMIT.toLong())
        val json = JSONObject(manifest.readText())
        SteeringSummary(vectorHash, json.getInt("layer_index"), json.optString("direction"))
    }.getOrNull()

    /** [expectedModelHash], when known, is the model the import was started from; a vector for another file is refused. */
    fun importVector(context: Context, uris: List<Uri>, expectedModelHash: String?, cancelled: () -> Boolean): SteeringProfile {
        require(uris.size == 2 && uris.distinct().size == 2) { context.getString(R.string.select_the_vector_gguf_and_its_matching_gguf_json_file_together) }
        val staging = File(root(context), ".import-${UUID.randomUUID()}").apply { check(mkdir()) }
        try {
            val files = uris.mapIndexed { index, uri ->
                File(staging, "input-$index").also { target ->
                    requireNotNull(context.contentResolver.openInputStream(uri)) { context.getString(R.string.cannot_open_the_selected_file) }.use { input ->
                        target.outputStream().use { output ->
                            val buffer = ByteArray(64 * 1024)
                            var total = 0
                            while (true) {
                                check(!cancelled()) { context.getString(R.string.steering_import_cancelled) }
                                val count = input.read(buffer)
                                if (count < 0) break
                                total += count
                                require(total <= VECTOR_LIMIT) { context.getString(R.string.choose_a_steering_vector_not_model_weights_8_mib_limit) }
                                output.write(buffer, 0, count)
                            }
                        }
                    }
                }
            }
            val vectors = files.filter { file -> file.inputStream().use {
                val magic = ByteArray(4)
                it.read(magic) == 4 && magic.decodeToString() == "GGUF"
            } }
            require(vectors.size == 1) { context.getString(R.string.select_exactly_one_gguf_vector_and_one_json_metadata_file) }
            val vector = vectors.single()
            val manifest = files.single { it != vector }
            require(manifest.length() in 1..METADATA_LIMIT.toLong()) { context.getString(R.string.steering_metadata_exceeds_1_mib) }
            val profile = SteeringProfile(NativeBridge.inspectSteering(vector.absolutePath.toByteArray(), manifest.readBytes()).decodeToString())
            require(expectedModelHash == null || profile.modelHash == expectedModelHash) {
                context.getString(R.string.this_vector_was_made_for_a_different_model_file)
            }
            check(!cancelled()) { context.getString(R.string.steering_import_cancelled) }
            check(vector.renameTo(File(staging, "vector.gguf"))) { context.getString(R.string.cannot_stage_vector) }
            check(manifest.renameTo(File(staging, "metadata.json"))) { context.getString(R.string.cannot_stage_metadata) }
            val model = modelDirectory(context, profile.modelHash).apply { mkdirs() }
            val destination = File(model, profile.vectorHash)
            if (destination.exists() && runCatching {
                    NativeBridge.inspectSteering(File(destination, "vector.gguf").absolutePath.toByteArray(),
                        File(destination, "metadata.json").readBytes())
                }.isFailure) {
                check(destination.renameTo(File(model, ".invalid-${UUID.randomUUID()}"))) { context.getString(R.string.cannot_preserve_damaged_steering_files) }
            }
            if (!destination.exists()) check(staging.renameTo(destination)) { context.getString(R.string.cannot_publish_steering_files) }
            // Verify an existing identical-hash package too; never silently reuse damaged private files.
            NativeBridge.inspectSteering(File(destination, "vector.gguf").absolutePath.toByteArray(),
                File(destination, "metadata.json").readBytes())
            val index = AtomicFile(File(model, "current.json"))
            val output = index.startWrite()
            try {
                output.write(JSONObject().put("vector_sha256", profile.vectorHash).toString().toByteArray())
                index.finishWrite(output)
            } catch (error: Exception) {
                index.failWrite(output)
                throw error
            }
            return profile
        } finally {
            staging.deleteRecursively()
        }
    }

    fun remove(context: Context, modelHash: String) {
        // Removing the association leaves the small verified package available for a later re-import.
        AtomicFile(File(modelDirectory(context, modelHash), "current.json")).delete()
    }
}
