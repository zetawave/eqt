package org.equity.app

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors

enum class AttachmentKind { Image, Document }

/** A file attached to a user message, kept in its chat's private media folder. */
data class Attachment(
    val kind: AttachmentKind,
    val name: String,
    /** File name in the media folder: a normalized JPEG for images, extracted UTF-8 text for documents. */
    val file: String,
    val mime: String,
    val bytes: Long,
)

data class Message(
    val id: Long,
    val fromUser: Boolean,
    val text: String,
    val attachments: List<Attachment> = emptyList(),
    val streaming: Boolean = false,
    val stats: String? = null,
    val failed: Boolean = false,
    /** Stopped before the end; shown, but the turn is not sent back to the model as context. */
    val stopped: Boolean = false,
)

data class Chat(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val messages: List<Message>,
) {
    val nextMessageId get() = (messages.maxOfOrNull { it.id } ?: 0L) + 1
}

data class ChatSummary(val id: String, val title: String, val updatedAt: Long)

data class ChatsState(
    /** Most recently updated first. */
    val summaries: List<ChatSummary> = emptyList(),
    /** Null while the user is on a new chat that has no messages yet. */
    val currentId: String? = null,
    /** The current chat and the chat being answered, if different. */
    val loaded: Map<String, Chat> = emptyMap(),
) {
    val current: Chat? get() = currentId?.let { loaded[it] }
}

/**
 * Conversations in app-private storage (`files/chats/<id>/chat.json` plus a `media` folder), excluded from
 * backups by data_extraction_rules. State changes are immediate; disk writes run in order on one IO thread.
 */
object ChatStore {
    private val flow = MutableStateFlow(ChatsState())
    val state: StateFlow<ChatsState> = flow.asStateFlow()
    private val io = Executors.newSingleThreadExecutor()
    private lateinit var root: File

    fun init(context: Context) {
        root = File(context.filesDir, "chats").apply { mkdirs() }
        // Attachments picked but never sent.
        staging.deleteRecursively()
        flow.value = ChatsState(summaries = readSummaries())
    }

    val staging: File get() = File(root, ".staging")
    private fun directory(id: String) = File(root, id)
    fun media(chatId: String) = File(directory(chatId), "media")
    fun file(chatId: String, attachment: Attachment) = File(media(chatId), attachment.file)

    fun newChat() = flow.update { it.copy(currentId = null, loaded = it.loaded.filterKeys(::busy)) }

    fun open(id: String) {
        val chat = flow.value.loaded[id] ?: read(id) ?: return
        flow.update { it.copy(currentId = id, loaded = it.loaded.filterKeys(::busy) + (id to chat)) }
    }

    fun delete(id: String) {
        flow.update { s ->
            s.copy(summaries = s.summaries.filter { it.id != id }, loaded = s.loaded - id,
                currentId = s.currentId.takeIf { it != id })
        }
        val summaries = flow.value.summaries
        io.execute {
            directory(id).deleteRecursively()
            writeIndex(summaries)
        }
    }

    /** Appends a user message and its pending answer, creating the chat when [chatId] is null. */
    fun startTurn(chatId: String?, text: String, attachments: List<Attachment>, now: Long = System.currentTimeMillis()): Pair<Chat, Long> {
        val existing = chatId?.let { flow.value.loaded[it] ?: read(it) }
        val chat = existing ?: Chat(UUID.randomUUID().toString(), title(text, attachments), now, now, emptyList())
        val userId = chat.nextMessageId
        // Attachments move from staging into the chat before the request refers to them.
        val media = media(chat.id).apply { mkdirs() }
        for (attachment in attachments) {
            val source = File(staging, attachment.file)
            if (!source.renameTo(File(media, attachment.file))) {
                source.copyTo(File(media, attachment.file), overwrite = true)
                source.delete()
            }
        }
        val next = chat.copy(updatedAt = now, messages = chat.messages +
            Message(userId, true, text, attachments) + Message(userId + 1, false, "", streaming = true))
        put(next, persist = true)
        // The turn was sent from the chat on screen; a new chat becomes the current one.
        flow.update { if (it.currentId == chatId) it.copy(currentId = next.id) else it }
        return next to userId + 1
    }

    fun updateMessage(chatId: String, messageId: Long, persist: Boolean, change: (Message) -> Message) {
        val chat = flow.value.loaded[chatId] ?: return
        put(chat.copy(messages = chat.messages.map { if (it.id == messageId) change(it) else it }), persist)
    }

    private fun busy(id: String) = EngineController.state.value.generation?.chatId == id

    private fun put(chat: Chat, persist: Boolean) {
        flow.update { s ->
            val summary = ChatSummary(chat.id, chat.title, chat.updatedAt)
            s.copy(loaded = s.loaded + (chat.id to chat),
                summaries = (listOf(summary) + s.summaries.filter { it.id != chat.id }).sortedByDescending { it.updatedAt })
        }
        if (persist) {
            val summaries = flow.value.summaries
            io.execute {
                // Deleted while an answer was still being written.
                if (summaries.none { it.id == chat.id }) return@execute
                directory(chat.id).mkdirs()
                atomicWrite(File(directory(chat.id), "chat.json"), encode(chat).toString())
                writeIndex(summaries)
            }
        }
    }

    private fun title(text: String, attachments: List<Attachment>): String {
        val line = text.lineSequence().map { it.trim() }.firstOrNull { it.isNotEmpty() }
        val title = line ?: attachments.firstOrNull()?.name ?: "New chat"
        return if (title.length <= 60) title else title.take(57).trimEnd() + "…"
    }

    private fun read(id: String): Chat? {
        val file = File(directory(id), "chat.json")
        val chat = runCatching { decode(JSONObject(file.readText())) }.getOrNull() ?: return null
        // A streaming answer on disk means the process ended while writing it.
        return chat.copy(messages = chat.messages.map {
            if (it.streaming) it.copy(streaming = false, failed = true, text = it.text.ifEmpty { "Interrupted." }) else it
        })
    }

    private fun readSummaries(): List<ChatSummary> {
        val index = File(root, "index.json")
        val listed = runCatching {
            val array = JSONArray(index.readText())
            (0 until array.length()).map { i ->
                array.getJSONObject(i).let { ChatSummary(it.getString("id"), it.getString("title"), it.getLong("updated")) }
            }
        }.getOrNull()
        if (listed != null && listed.all { File(directory(it.id), "chat.json").isFile }) return listed
        // Missing or stale index: rebuild it from the chats on disk.
        return root.listFiles().orEmpty().filter { it.isDirectory && !it.name.startsWith(".") }
            .mapNotNull { dir -> read(dir.name)?.let { ChatSummary(it.id, it.title, it.updatedAt) } }
            .sortedByDescending { it.updatedAt }
            .also { writeIndex(it) }
    }

    private fun writeIndex(summaries: List<ChatSummary>) {
        val array = JSONArray()
        summaries.forEach { array.put(JSONObject().put("id", it.id).put("title", it.title).put("updated", it.updatedAt)) }
        atomicWrite(File(root, "index.json"), array.toString())
    }

    private fun atomicWrite(target: File, text: String) {
        val temporary = File(target.parentFile, target.name + ".tmp")
        temporary.writeText(text)
        if (!temporary.renameTo(target)) {
            target.delete()
            temporary.renameTo(target)
        }
    }

    private fun encode(chat: Chat): JSONObject {
        val messages = JSONArray()
        for (m in chat.messages) {
            val attachments = JSONArray()
            m.attachments.forEach {
                attachments.put(JSONObject().put("kind", it.kind.name).put("name", it.name).put("file", it.file)
                    .put("mime", it.mime).put("bytes", it.bytes))
            }
            messages.put(JSONObject().put("id", m.id).put("user", m.fromUser).put("text", m.text)
                .put("attachments", attachments).put("streaming", m.streaming).put("stats", m.stats ?: JSONObject.NULL)
                .put("failed", m.failed).put("stopped", m.stopped))
        }
        return JSONObject().put("version", 1).put("id", chat.id).put("title", chat.title)
            .put("created", chat.createdAt).put("updated", chat.updatedAt).put("messages", messages)
    }

    private fun decode(json: JSONObject): Chat {
        val array = json.getJSONArray("messages")
        val messages = (0 until array.length()).map { i ->
            val m = array.getJSONObject(i)
            val list = m.optJSONArray("attachments") ?: JSONArray()
            Message(
                id = m.getLong("id"), fromUser = m.getBoolean("user"), text = m.getString("text"),
                attachments = (0 until list.length()).map { j ->
                    val a = list.getJSONObject(j)
                    Attachment(AttachmentKind.valueOf(a.getString("kind")), a.getString("name"), a.getString("file"),
                        a.getString("mime"), a.getLong("bytes"))
                },
                streaming = m.optBoolean("streaming"), stats = if (m.isNull("stats")) null else m.getString("stats"),
                failed = m.optBoolean("failed"), stopped = m.optBoolean("stopped"),
            )
        }
        return Chat(json.getString("id"), json.getString("title"), json.getLong("created"), json.getLong("updated"), messages)
    }
}
