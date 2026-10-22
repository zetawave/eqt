package org.equity.app

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Keeps Equity running while a model loads or an answer is written with the app in the background: a foreground
 * service with an ongoing notification (progress and Stop) and a partial wake lock, so the CPU does not suspend
 * when the screen turns off. It stops itself as soon as the engine is idle. Thermal limits stay with the system.
 */
class InferenceService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var wakeLock: PowerManager.WakeLock? = null
    private var lastNotified = 0L
    // Stopping before the first startForeground() would make Android treat the start as failed.
    private var foreground = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Notifications.ensureChannels(this)
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Equity:inference").apply {
                setReferenceCounted(false)
                // Upper bound in case the engine never returns; a normal answer releases it much earlier.
                acquire(WAKE_LOCK_LIMIT_MS)
            }
        scope.launch {
            EngineController.state.collect { state ->
                if (!state.busy) {
                    if (foreground) stop()
                    return@collect
                }
                val now = SystemClock.elapsedRealtime()
                // The system drops updates above a few per second; tokens arrive faster than that.
                if (now - lastNotified >= 1000) {
                    lastNotified = now
                    Notifications.post(this@InferenceService, Notifications.PROGRESS_ID, Notifications.progress(this@InferenceService, state))
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) EngineController.stop()
        // Every startForegroundService() must be answered with startForeground(), even when already idle.
        val type = if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE else 0
        ServiceCompat.startForeground(this, Notifications.PROGRESS_ID,
            Notifications.progress(this, EngineController.state.value), type)
        foreground = true
        if (!EngineController.state.value.busy) stop()
        return START_NOT_STICKY
    }

    private fun stop() {
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        scope.cancel()
        wakeLock?.takeIf { it.isHeld }?.release()
        super.onDestroy()
    }

    companion object {
        const val ACTION_STOP = "org.equity.app.action.STOP"
        private const val WAKE_LOCK_LIMIT_MS = 3 * 60 * 60 * 1000L

        /** Call while Equity is in the foreground (a user action); Android refuses background starts. */
        fun start(context: Context) {
            runCatching { ContextCompat.startForegroundService(context, Intent(context, InferenceService::class.java)) }
        }
    }
}

object Notifications {
    const val PROGRESS_ID = 4001
    private const val FINISHED_ID_BASE = 4200
    private const val CHANNEL_PROGRESS = "inference"
    private const val CHANNEL_FINISHED = "answers"
    const val EXTRA_CHAT = "org.equity.app.extra.CHAT"

    fun ensureChannels(context: Context) {
        val manager = context.getSystemService(NotificationManager::class.java)
        run {
            manager.createNotificationChannel(NotificationChannel(CHANNEL_PROGRESS, context.getString(R.string.answers_in_progress),
                NotificationManager.IMPORTANCE_LOW).apply { description = context.getString(R.string.shown_while_equity_loads_a_model_or_writes_an_answer) })
        }
        run {
            manager.createNotificationChannel(NotificationChannel(CHANNEL_FINISHED, context.getString(R.string.finished_answers),
                NotificationManager.IMPORTANCE_DEFAULT).apply { description = context.getString(R.string.an_answer_finished_while_equity_was_in_the_background) })
        }
    }

    fun progress(context: Context, state: EngineState): android.app.Notification {
        val generation = state.generation
        val builder = NotificationCompat.Builder(context, CHANNEL_PROGRESS)
            .setSmallIcon(R.drawable.ic_notification)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setContentIntent(openApp(context, generation?.chatId))
            .addAction(0, context.getString(R.string.stop), PendingIntent.getService(context, 1,
                Intent(context, InferenceService::class.java).setAction(InferenceService.ACTION_STOP), PendingIntent.FLAG_IMMUTABLE))
        when {
            state.phase == Phase.Loading -> builder.setContentTitle(context.getString(R.string.loading_1_s, state.modelName ?: context.getString(R.string.model)))
                .setContentText(state.status)
                .setProgress(100, state.loadPercent ?: 0, state.loadPercent == null)
            generation != null && generation.tokens > 0 && generation.thinking -> {
                val seconds = (SystemClock.elapsedRealtime() - generation.startedAt) / 1000.0
                builder.setContentTitle(context.getString(R.string.equity_is_thinking))
                    .setContentText(context.getString(R.string.value_1_d_tokens_2_s, generation.tokens, elapsed(seconds)))
                    .setProgress(0, 0, true)
            }
            generation != null && generation.tokens > 0 -> {
                val seconds = (SystemClock.elapsedRealtime() - generation.startedAt) / 1000.0
                builder.setContentTitle(if (generation.hasVisibleText) context.getString(R.string.equity_is_writing_an_answer) else context.getString(R.string.equity_is_generating_tokens))
                    .setContentText(context.getString(R.string.value_1_d_tokens_2_s_3_s, generation.tokens, if (generation.hasVisibleText) "" else context.getString(R.string.no_visible_text_yet), elapsed(seconds)))
                    .setProgress(0, 0, true)
            }
            else -> builder.setContentTitle(context.getString(R.string.equity_is_reading_your_message)).setContentText(state.status).setProgress(0, 0, true)
        }
        return builder.build()
    }

    /** Posts a notification for an answer that finished while Equity was not on screen. */
    fun answerFinished(context: Context, chatId: String, title: String, text: String, failed: Boolean, stopped: Boolean = false) {
        val heading = when {
            failed -> context.getString(R.string.answer_failed)
            stopped -> context.getString(R.string.answer_stopped)
            !text.hasVisibleContent() -> context.getString(R.string.no_visible_answer)
            else -> context.getString(R.string.answer_ready)
        }
        val preview = if (text.hasVisibleContent()) text else context.getString(R.string.the_model_finished_without_producing_visible_text)
        val notification = NotificationCompat.Builder(context, CHANNEL_FINISHED)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("$heading · $title")
            .setContentText(preview.take(200))
            .setStyle(NotificationCompat.BigTextStyle().bigText(preview.take(1000)))
            .setAutoCancel(true)
            .setCategory(NotificationCompat.CATEGORY_MESSAGE)
            .setContentIntent(openApp(context, chatId))
            .build()
        post(context, FINISHED_ID_BASE + (chatId.hashCode() and 0xffff), notification)
    }

    fun post(context: Context, id: Int, notification: android.app.Notification) {
        val allowed = Build.VERSION.SDK_INT < 33 ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        if (allowed) runCatching { NotificationManagerCompat.from(context).notify(id, notification) }
    }

    private fun openApp(context: Context, chatId: String?): PendingIntent {
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        if (chatId != null) intent.putExtra(EXTRA_CHAT, chatId)
        return PendingIntent.getActivity(context, chatId?.hashCode() ?: 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }

    private fun elapsed(seconds: Double) =
        if (seconds < 60) "%.0f s".format(Locale.US, seconds) else "%d min %02d s".format(Locale.US, (seconds / 60).toInt(), (seconds % 60).toInt())
}
