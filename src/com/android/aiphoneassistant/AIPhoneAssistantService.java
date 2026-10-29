/*
 * Copyright (C) 2026 The AgentVisOS Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.aiphoneassistant;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.LauncherApps;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.graphics.PointF;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.os.Binder;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.Display;
import android.view.KeyEvent;
import android.widget.Toast;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Main AI Phone Assistant Service.
 *
 * Owns one task lifecycle at a time:
 *   pending (set via ChatActivity / Settings / boot) → beginExecution():
 *   show the interrupt bar + live foreground notification, then run the
 *   see → think → act loop on a background capture thread:
 *     1. Captures the screen (WindowManager captureDisplay, with a mirror
 *        VirtualDisplay fallback).
 *     2. When {@link FrameChangeDetector} decides a settled, meaningful change
 *        happened, sends the frame to the configured LLM with the task.
 *     3. Injects the returned action via InputManager, recording each action
 *        node (description + screenshot) in the {@link ChatStore}.
 *   → end paths, all of which deactivate the service again:
 *     - the model replies done (summary) → result notification,
 *     - the user interrupts (triple-tap on the interrupt bar, confirm dialog),
 *     - the stale-task watchdog exhausts / repeated LLM errors → stalled.
 *   A new task arriving later re-activates the service.
 *
 * While executing, three quick taps on the interrupt bar PAUSE the loop and
 * open a confirm dialog: continue resumes (the frame detector re-baselines),
 * stop takes the interrupt path. During each input injection the interrupt bar
 * is briefly made non-touchable so the injected events reach the app.
 */
public class AIPhoneAssistantService extends Service {
    private static final String TAG = "AIPhoneAssistant";

    public static final String ACTION_ACTIVATE =
            "com.android.aiphoneassistant.action.ACTIVATE";
    public static final String ACTION_INTERRUPT =
            "com.android.aiphoneassistant.action.INTERRUPT";

    /** Broadcast on every task-state change; ChatActivity listens and refreshes. */
    public static final String BROADCAST_STATE_CHANGED =
            "com.android.aiphoneassistant.STATE_CHANGED";
    public static final String EXTRA_STATE = "state";
    public static final String EXTRA_TASK = "task";
    public static final String EXTRA_CONV = "conv";

    private static final String NOTIFICATION_CHANNEL_ID = "ai_phone_assistant";
    private static final int NOTIFICATION_ID = 1001;
    private static final String RESULT_CHANNEL_ID = "ai_task_result";
    private static final int RESULT_NOTIFICATION_ID = 1002;

    // How long to leave the "acting" overlay up after injecting an action
    private static final long ACT_HIDE_DELAY_MS = 400;
    // Keep the interrupt bar non-touchable this long after the last injected
    // event, so the injection has been dispatched before touches are captured
    // by the bar again.
    private static final long BAR_RETOUCH_DELAY_MS = 120;
    // Only log the first of a run of consecutive capture failures
    private static final int CAPTURE_FAILURE_LOG_THRESHOLD = 3;
    // Give up (stalled) after this many consecutive LLM request failures
    private static final int MAX_CONSECUTIVE_LLM_ERRORS = 5;
    // Hard bound for one LLM round. Slow-dripping responses can evade the
    // HTTP read timeout; without this the loop would freeze forever with
    // mAnalyzing stuck true (the watchdog lives inside the skipped block).
    private static final long ANALYZE_TIMEOUT_MS = 300000;
    // How many recent actions are included in the prompt
    private static final int PROMPT_ACTION_HISTORY = 8;
    // How many device observations (list_apps / open_app results) to keep
    private static final int PROMPT_OBSERVATION_LIMIT = 6;
    // Hard cap on one observation: a list_apps answer is naturally long
    private static final int OBSERVATION_MAX_CHARS = 2400;
    // Settle time after launching an app before the next capture
    private static final long APP_LAUNCH_SETTLE_MS = 900;
    private static final int TASK_TITLE_MAX_LEN = 40;

    // Stale-task watchdog: when a task is executing but nothing has been
    // escalated for STALE_TASK_REANALYZE_MS (e.g. the model returned the
    // screen to an already-analyzed state), force a re-analysis so the model
    // gets another chance. Bounded by MAX_FORCED_REANALYSES consecutive
    // forced rounds: if the last forced round still yields no action the
    // task is declared stalled and the service deactivates. A natural
    // (change-driven) escalation refills the budget.
    private static final long STALE_TASK_REANALYZE_MS = 20000;
    private static final int MAX_FORCED_REANALYSES = 3;
    private long mLastEscalationAtMs;
    private int mForcedReanalyses = 0;
    // True when the current analysis was triggered by the watchdog (read on
    // the capture thread only).
    private boolean mForcedRound = false;

    private final IBinder mBinder = new AIPhoneAssistantBinder();
    private final AtomicBoolean mIsRunning = new AtomicBoolean(false);
    private final AtomicBoolean mAnalyzing = new AtomicBoolean(false);
    private final AtomicBoolean mPaused = new AtomicBoolean(false);
    // Set when the user confirms the stop dialog; makes the loop and any
    // in-flight work wind down at the next checkpoint.
    private volatile boolean mInterrupted = false;

    private static final AtomicBoolean sRunning = new AtomicBoolean(false);

    private HandlerThread mCaptureThread;
    private Handler mCaptureHandler;
    private Handler mMainHandler;

    // Screen capture components (fallback path only)
    private ImageReader mImageReader;
    private VirtualDisplay mVirtualDisplay;

    // Subsystems
    private ScreenCaptureService mScreenCaptureService;
    private InputInjectionService mInputInjectionService;
    private LLMClientService mLLMClientService;
    private FrameChangeDetector mFrameDetector;
    private OverlayController mOverlayController;
    private ChatStore mChatStore;

    // Task lifecycle (capture thread unless noted)
    private volatile String mCurrentTask;
    private volatile long mConvId;
    private Bitmap mCurrentFrame;
    private final List<String> mActionHistory = new ArrayList<>();
    /**
     * Non-visual results (list_apps catalogue slices, open_app failures) that
     * the next prompt must see. Bounded: a full app catalogue would consume
     * the whole context for no benefit.
     */
    private final List<String> mObservations = new ArrayList<>();
    private int mConsecutiveLlmErrors = 0;
    private int mConsecutiveCaptureFailures = 0;
    // Generation counter for the in-flight analysis; bumped whenever a round
    // ends so a late timeout callback knows it is stale.
    private int mAnalyzeGen = 0;
    /** Force the next captured frame to be analysed even without a visual change. */
    private volatile boolean mForceNextAnalysis = false;

    /** Cheap liveness check for the UI (ChatActivity header). */
    public static boolean isRunning() {
        return sRunning.get();
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG, "AI Phone Assistant Service created");
        sRunning.set(true);

        mMainHandler = new Handler(Looper.getMainLooper());

        mCaptureThread = new HandlerThread("AIPhoneAssistant-Capture");
        mCaptureThread.start();
        mCaptureHandler = new Handler(mCaptureThread.getLooper());

        mScreenCaptureService = new ScreenCaptureService(this);
        mInputInjectionService = new InputInjectionService(this);
        mLLMClientService = new LLMClientService(this);
        mFrameDetector = new FrameChangeDetector();
        mOverlayController = new OverlayController(this, mMainHandler);
        mChatStore = new ChatStore(this);

        createNotificationChannels();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent != null && intent.getAction() != null
                ? intent.getAction() : ACTION_ACTIVATE;
        Log.i(TAG, "AI Phone Assistant Service started (action=" + action + ")");
        startForeground(NOTIFICATION_ID, buildFgNotification(
                getString(R.string.notification_starting)));

        if (ACTION_INTERRUPT.equals(action)) {
            if (mIsRunning.get()) {
                beginInterrupt();
            } else {
                stopSelf();
            }
            return START_STICKY;
        }

        SharedPreferences prefs = LLMClientService.getPrefs(this);
        if (!prefs.getBoolean(LLMClientService.KEY_ENABLED, false)) {
            Log.i(TAG, "Assistant disabled; not activating");
            stopSelf();
            return START_NOT_STICKY;
        }

        if (mIsRunning.compareAndSet(false, true)) {
            mLastEscalationAtMs = System.currentTimeMillis();
            mForcedReanalyses = 0;
            mCaptureHandler.post(() -> mScreenCaptureService.initialize());
            mCaptureHandler.post(mCaptureRunnable);
        }
        mCaptureHandler.post(this::beginExecution);

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        Log.i(TAG, "AI Phone Assistant Service stopping");
        sRunning.set(false);
        mIsRunning.set(false);
        mInterrupted = true;

        // If the service is torn down mid-task (user flipped the master
        // switch, system killed us), record the stop in the chat so the UI
        // does not keep claiming a task is running.
        String state = ChatStore.getTaskState(this);
        if (ChatStore.STATE_EXECUTING.equals(state) || ChatStore.STATE_PAUSED.equals(state)) {
            ChatStore.setTaskState(this, ChatStore.STATE_STOPPED);
            if (mChatStore != null && mConvId > 0) {
                mChatStore.appendStatus(mConvId, getString(R.string.chat_status_stopped));
                mChatStore.updateConversation(mConvId, ChatStore.STATE_STOPPED,
                        getString(R.string.chat_status_stopped));
            }
        }

        if (mCaptureHandler != null) {
            mCaptureHandler.removeCallbacksAndMessages(null);
        }
        releaseScreenCapture();

        if (mLLMClientService != null) {
            mLLMClientService.cancelInFlight();
        }
        if (mInputInjectionService != null) {
            mInputInjectionService.setAborted(true);
        }
        if (mScreenCaptureService != null) {
            mScreenCaptureService.release();
        }
        if (mInputInjectionService != null) {
            mInputInjectionService.release();
        }
        if (mLLMClientService != null) {
            mLLMClientService.release();
            LLMClientService.unloadLocalModel(this);
        }
        if (mOverlayController != null) {
            mOverlayController.release();
        }

        if (mCaptureThread != null) {
            mCaptureThread.quitSafely();
            try {
                mCaptureThread.join();
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while stopping capture thread", e);
            }
        }

        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    // ------------------------------------------------------------------
    // Task lifecycle
    // ------------------------------------------------------------------

    /** Called on the capture thread: pick up the pending task and activate. */
    private void beginExecution() {
        String task = ChatStore.getTask(this);
        if (task.isEmpty()) {
            Log.i(TAG, "No task configured; deactivating");
            stopSelf();
            return;
        }
        String state = ChatStore.getTaskState(this);
        if (!ChatStore.STATE_PENDING.equals(state)
                && !ChatStore.STATE_EXECUTING.equals(state)
                && !ChatStore.STATE_PAUSED.equals(state)) {
            // Sticky restart or stray start with a finished/old task: the
            // task must be (re)armed via a new conversation from the UI.
            Log.i(TAG, "Task state is '" + state + "'; not re-executing old task");
            stopSelf();
            return;
        }

        mConvId = ChatStore.getCurrentConvId(this);
        if (mConvId <= 0 || mChatStore.getConversation(mConvId) == null) {
            // Fallback for start paths that did not create a conversation
            // (should not happen; keeps the transcript complete anyway).
            mConvId = mChatStore.createConversation(this, task);
        }

        mCurrentTask = task;
        ChatStore.setTaskState(this, ChatStore.STATE_EXECUTING);
        mChatStore.updateConversation(mConvId, ChatStore.STATE_EXECUTING, task);
        mInterrupted = false;
        mPaused.set(false);
        mInputInjectionService.setAborted(false);
        mActionHistory.clear();
        mObservations.clear();
        // Enumerating a few hundred packages costs a second; get it off the
        // capture thread before the model first asks for apps.
        AppCatalog.warmUp(this);
        mForcedReanalyses = 0;
        mForcedRound = false;
        mConsecutiveLlmErrors = 0;
        mConsecutiveCaptureFailures = 0;
        mLastEscalationAtMs = System.currentTimeMillis();

        mChatStore.appendStatus(mConvId, getString(R.string.chat_status_started_format, shorten(task)));
        broadcastState(ChatStore.STATE_EXECUTING);
        updateFgNotification(getString(R.string.notification_observing));
        mOverlayController.showInterruptBar(mInterruptBarListener);
    }

    /**
     * Main loop: capture at the configured interval and escalate to the LLM
     * when the frame detector decides the change is worth analyzing. Paused
     * while the interrupt confirm dialog is up.
     */
    private final Runnable mCaptureRunnable = new Runnable() {
        @Override
        public void run() {
            if (!mIsRunning.get()) {
                return;
            }

            int intervalMs = LLMClientService.getPrefs(AIPhoneAssistantService.this)
                    .getInt(LLMClientService.KEY_CAPTURE_INTERVAL,
                            LLMClientService.DEFAULT_CAPTURE_INTERVAL_MS);

            if (mPaused.get() || mInterrupted) {
                mCaptureHandler.postDelayed(this, Math.max(intervalMs, 100));
                return;
            }

            boolean configured = LLMClientService.isConfigured(AIPhoneAssistantService.this)
                    && mCurrentTask != null && !mCurrentTask.trim().isEmpty();

            try {
                boolean overlayVisible = mOverlayController != null
                        && mOverlayController.isVisible();
                // Skip capture while an LLM request is in flight or the overlay is up:
                // its own pixels would feed the change detector and could self-trigger.
                if (configured && !overlayVisible && mAnalyzing.compareAndSet(false, true)) {
                    Bitmap screenshot = captureScreen();

                    if (screenshot != null) {
                        mConsecutiveCaptureFailures = 0;
                        long now = System.currentTimeMillis();
                        boolean stale = now - mLastEscalationAtMs >= STALE_TASK_REANALYZE_MS;
                        // A device action (open_app / list_apps) may leave the
                        // screen unchanged but always produces new information
                        // for the model, so analyse the very next frame.
                        boolean forceNow = mForceNextAnalysis;
                        boolean allowForce = forceNow
                                || (stale && mForcedReanalyses < MAX_FORCED_REANALYSES);
                        boolean escalate;
                        if (allowForce) {
                            escalate = mFrameDetector.forceAnalyze(screenshot);
                            if (escalate) {
                                if (forceNow) {
                                    mForceNextAnalysis = false;
                                    mForcedReanalyses = 0;
                                    mForcedRound = false;
                                } else {
                                    mForcedReanalyses++;
                                    mForcedRound = true;
                                    Log.i(TAG, "Stale-task watchdog: forcing re-analysis (#"
                                            + mForcedReanalyses + ", no escalation for "
                                            + sinceEscalationSeconds(mLastEscalationAtMs, now)
                                            + "s)");
                                }
                            }
                        } else {
                            escalate = mFrameDetector.shouldAnalyze(screenshot);
                            if (escalate) {
                                // a natural change refills the watchdog budget
                                mForcedReanalyses = 0;
                                mForcedRound = false;
                            }
                        }
                        if (escalate) {
                            analyzeWithLLM(screenshot);
                        } else {
                            mAnalyzing.set(false);
                        }
                    } else {
                        mAnalyzing.set(false);
                        if (++mConsecutiveCaptureFailures == CAPTURE_FAILURE_LOG_THRESHOLD) {
                            Log.e(TAG, "Screen capture keeps failing ("
                                    + mConsecutiveCaptureFailures + " attempts)");
                        }
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Error during screen capture", e);
                mAnalyzing.set(false);
            }

            mCaptureHandler.postDelayed(this, Math.max(intervalMs, 100));
        }
    };

    private static long sinceEscalationSeconds(long lastEscalationAtMs, long now) {
        return Math.max(0, (now - lastEscalationAtMs) / 1000);
    }

    /**
     * Send the frame to the LLM and act on the reply. The mAnalyzing flag is
     * already set by the caller; it is cleared on completion (including
     * errors).
     */
    private void analyzeWithLLM(Bitmap screenshot) {
        mLastEscalationAtMs = System.currentTimeMillis();
        mCurrentFrame = screenshot;
        if (mOverlayController != null) {
            mOverlayController.showThinking();
        }
        updateFgNotification(getString(R.string.notification_thinking));

        final int gen = ++mAnalyzeGen;
        mCaptureHandler.removeCallbacks(mAnalyzeTimeout);
        mCaptureHandler.postDelayed(mAnalyzeTimeout, ANALYZE_TIMEOUT_MS);

        // captureDisplay returns the screen at native resolution, so when the
        // display metrics are unknown the screenshot's own size is the
        // correct mapping target (scale = 1).
        int screenW = mScreenCaptureService.getDisplayWidth();
        int screenH = mScreenCaptureService.getDisplayHeight();
        if (screenW <= 0 || screenH <= 0) {
            screenW = screenshot.getWidth();
            screenH = screenshot.getHeight();
        }

        mLLMClientService.analyzeScreen(screenshot,
                screenW,
                screenH,
                mCurrentTask,
                recentActionHistory(),
                recentObservations(),
                new LLMClientService.LLMCallback() {
            @Override
            public void onDecision(LLMClientService.LLMDecision decision) {
                // Inject on the capture (background) thread, never the UI thread:
                // the injectors block with Thread.sleep and would freeze the UI.
                mCaptureHandler.post(() -> {
                    endAnalyzeRound();
                    handleDecision(decision);
                });
            }

            @Override
            public void onInterrupted() {
                mCaptureHandler.post(() -> {
                    endAnalyzeRound();
                    mAnalyzing.set(false);
                    if (mOverlayController != null) {
                        mOverlayController.hide();
                    }
                });
            }

            @Override
            public void onError(String error) {
                Log.e(TAG, "LLM analysis error: " + error);
                mCaptureHandler.post(() -> {
                    endAnalyzeRound();
                    handleLlmError();
                });
            }
        });
    }

    private List<String> recentActionHistory() {
        int from = Math.max(0, mActionHistory.size() - PROMPT_ACTION_HISTORY);
        return new ArrayList<>(mActionHistory.subList(from, mActionHistory.size()));
    }

    /**
     * Execute device-level actions: {@code open_app} / {@code open_shortcut} /
     * {@code list_apps}. They are not input injections; the outcome rides into
     * the next prompt as an observation. Runs on the capture thread; the
     * catalogue is usually warm because the task start kicked a background
     * build.
     */
    private void handleDeviceAction(InputInjectionService.InputAction action) {
        String desc;
        String observation;
        if (action.type == InputInjectionService.ActionType.LIST_APPS) {
            AppCatalog catalog = AppCatalog.get(this);
            String query = action.query == null ? "" : action.query.trim();
            // Long answers are useful for a broad query, but they must not
            // crowd out the action history in the prompt.
            int limit = query.isEmpty() ? 30 : 25;
            observation = catalog.describeForPrompt(query, limit);
            desc = getString(R.string.chat_action_list_apps_format,
                    query.isEmpty() ? "*" : query);
        } else if (action.type == InputInjectionService.ActionType.OPEN_SHORTCUT) {
            String requested = action.app == null ? "" : action.app.trim();
            String wanted = action.shortcut == null ? "" : action.shortcut.trim();
            AppCatalog catalog = AppCatalog.get(this);
            AppCatalog.App app = catalog.resolve(requested);
            String failure = null;
            String shortcutId = null;
            if (app == null) {
                failure = getString(R.string.observation_open_app_unknown_format, requested);
            } else {
                shortcutId = app.resolveShortcutId(wanted);
                if (shortcutId == null) {
                    failure = "No shortcut matching \"" + wanted + "\" in " + app.label;
                } else {
                    try {
                        LauncherApps launcherApps = getSystemService(LauncherApps.class);
                        if (launcherApps == null) {
                            failure = "LauncherApps service unavailable";
                        } else {
                            launcherApps.startShortcut(app.pkg, shortcutId, null, null,
                                    android.os.Process.myUserHandle());
                        }
                    } catch (Exception e) {
                        Log.w(TAG, "Shortcut launch failed: " + app.pkg + "/" + shortcutId, e);
                        failure = "Shortcut launch failed: " + e.getMessage();
                    }
                }
            }
            if (failure != null) {
                observation = failure;
                desc = "open_shortcut failed: " + requested + " / " + wanted;
            } else {
                observation = "Opened shortcut \"" + wanted + "\" (" + shortcutId + ") in "
                        + app.label;
                desc = "Opened " + app.label + " shortcut: " + wanted;
                mFrameDetector.notifyActionInjected();
                sleepQuietly(APP_LAUNCH_SETTLE_MS);
            }
        } else {
            String requested = action.app == null ? "" : action.app.trim();
            AppCatalog catalog = AppCatalog.get(this);
            AppCatalog.App app = catalog.resolve(requested);
            if (app == null) {
                observation = getString(R.string.observation_open_app_unknown_format, requested);
                desc = getString(R.string.chat_action_open_app_failed_format, requested);
            } else {
                Intent intent = AppCatalog.buildLaunchIntent(this, app, action.uri);
                String failure = null;
                if (intent == null) {
                    failure = getString(R.string.observation_open_app_no_intent_format, app.label);
                } else {
                    try {
                        startActivity(intent);
                    } catch (Exception e) {
                        Log.w(TAG, "Launch failed for " + app.pkg, e);
                        failure = getString(R.string.observation_open_app_failed_format,
                                app.label, String.valueOf(e.getMessage()));
                    }
                }
                if (failure != null) {
                    observation = failure;
                    desc = getString(R.string.chat_action_open_app_failed_format, requested);
                } else {
                    observation = getString(R.string.observation_open_app_ok_format,
                            app.label, app.pkg);
                    desc = action.uri == null || action.uri.isEmpty()
                            ? getString(R.string.chat_action_open_app_format, app.label)
                            : getString(R.string.chat_action_open_app_uri_format,
                                    app.label, action.uri);
                    // The window transition is a real change; give the target
                    // app time to draw before the next capture.
                    mFrameDetector.notifyActionInjected();
                    sleepQuietly(APP_LAUNCH_SETTLE_MS);
                }
            }
        }

        Log.i(TAG, "Device action: " + desc + " -> " + shorten(observation));
        addObservation(observation);
        mActionHistory.add(desc);
        mChatStore.appendAction(mConvId, desc, mCurrentFrame);
        mForcedReanalyses = 0;
        mForcedRound = false;
        // The observation is only useful if the model gets another round, and a
        // list_apps answer does not change the screen at all.
        mForceNextAnalysis = true;
        broadcastState(ChatStore.STATE_EXECUTING);
        updateFgNotification(getString(R.string.notification_acting_format, desc));
    }

    /** Device observations since the task started, oldest first. */
    private List<String> recentObservations() {
        return new ArrayList<>(mObservations);
    }

    /** Append one observation, keeping the ring bounded. */
    private void addObservation(String text) {
        if (text == null || text.trim().isEmpty()) {
            return;
        }
        String flat = text.trim();
        if (flat.length() > OBSERVATION_MAX_CHARS) {
            flat = flat.substring(0, OBSERVATION_MAX_CHARS) + "…";
        }
        mObservations.add(flat);
        while (mObservations.size() > PROMPT_OBSERVATION_LIMIT) {
            mObservations.remove(0);
        }
    }

    /** Mark the current analysis round as finished and cancel its timeout. */
    private void endAnalyzeRound() {
        mAnalyzeGen++;
        mCaptureHandler.removeCallbacks(mAnalyzeTimeout);
    }

    /** Abort the in-flight LLM request when one round exceeds the hard bound;
     *  a bit later, force-recover the loop in case even the cancel stalls. */
    private final Runnable mAnalyzeTimeout = new Runnable() {
        @Override
        public void run() {
            if (!mIsRunning.get()) {
                return;
            }
            if (mAnalyzing.get()) {
                Log.w(TAG, "LLM round exceeded " + ANALYZE_TIMEOUT_MS + " ms; aborting");
                mLLMClientService.cancelInFlight();
                mCaptureHandler.postDelayed(() -> {
                    if (mAnalyzing.get() && mIsRunning.get()) {
                        Log.w(TAG, "LLM round still stuck; force-recovering loop");
                        endAnalyzeRound();
                        mAnalyzing.set(false);
                    }
                }, ANALYZE_TIMEOUT_MS + 15000);
            }
        }
    };

    /** Called on the capture thread with the model's decision. */
    private void handleDecision(LLMClientService.LLMDecision decision) {
        try {
            // A decision (any kind) means the request succeeded; only
            // consecutive failures count towards the stall bound.
            mConsecutiveLlmErrors = 0;

            if (decision.kind == LLMClientService.LLMDecision.Kind.DONE) {
                String summary = decision.summary != null && !decision.summary.trim().isEmpty()
                        ? decision.summary.trim()
                        : getString(R.string.chat_result_default);
                endExecutionDone(summary);
                return;
            }

            if (decision.kind == LLMClientService.LLMDecision.Kind.NONE) {
                if (mForcedRound && mForcedReanalyses >= MAX_FORCED_REANALYSES) {
                    Log.i(TAG, "Watchdog exhausted with no model action; task stalled");
                    endExecutionStalled(null);
                }
                return;
            }

            // ACTION
            if (mInterrupted || mPaused.get()) {
                // Stop/resume was requested while the request was in flight:
                // skip injecting a stale decision. After a resume the frame
                // detector is re-baselined and the screen is re-analyzed.
                return;
            }

            // Device-level actions are not input injections: resolve them
            // against the app catalogue and let the outcome ride into the next
            // prompt as an observation.
            if (decision.action.type == InputInjectionService.ActionType.OPEN_APP
                    || decision.action.type == InputInjectionService.ActionType.OPEN_SHORTCUT
                    || decision.action.type == InputInjectionService.ActionType.LIST_APPS) {
                handleDeviceAction(decision.action);
                return;
            }

            PointF point = decision.action.getPrimaryPoint();
            if (mOverlayController != null && point != null) {
                mOverlayController.showActing(point.x, point.y);
            }
            String desc = describeAction(decision.action);
            updateFgNotification(getString(R.string.notification_acting_format, desc));

            // Let injected events reach the app: the interrupt bar must not
            // capture touches while we are injecting (it covers the middle of
            // the status bar strip).
            mOverlayController.setInterruptBarTouchable(false);
            boolean completed = mInputInjectionService.executeAction(decision.action);
            sleepQuietly(BAR_RETOUCH_DELAY_MS);
            mOverlayController.setInterruptBarTouchable(true);

            if (mInterrupted) {
                return;
            }
            if (!completed) {
                Log.i(TAG, "Action aborted by interrupt");
                return;
            }

            mFrameDetector.notifyActionInjected();
            // An executed action is real progress: refill the watchdog
            // budget so typing-heavy tasks (small on-screen deltas) are not
            // starved of re-analyses. Only forced rounds that yield NO
            // action consecutively may exhaust it into a stall.
            mForcedReanalyses = 0;
            mForcedRound = false;
            if (decision.action.type != InputInjectionService.ActionType.WAIT) {
                mActionHistory.add(decision.historyEntry != null
                        ? decision.historyEntry : desc);
                mChatStore.appendAction(mConvId, desc, mCurrentFrame);
                broadcastState(ChatStore.STATE_EXECUTING);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error executing action", e);
        } finally {
            mAnalyzing.set(false);
            // Let the acting marker finish, then hide and resume capture.
            if (mOverlayController != null && !mPaused.get() && !mInterrupted) {
                mOverlayController.hideDelayed(ACT_HIDE_DELAY_MS);
            } else if (mOverlayController != null) {
                mOverlayController.hide();
            }
        }
    }

    private void handleLlmError() {
        mAnalyzing.set(false);
        if (mOverlayController != null) {
            mOverlayController.hide();
        }
        if (mInterrupted || !mIsRunning.get()) {
            return;
        }
        // Repeated request failures must not leave a zombie loop running: if
        // the failure happened on the last watchdog round (no more forcing
        // will occur) or errors pile up back to back, end the task.
        if ((mForcedRound && mForcedReanalyses >= MAX_FORCED_REANALYSES)
                || mConsecutiveLlmErrors + 1 >= MAX_CONSECUTIVE_LLM_ERRORS) {
            endExecutionStalled(getString(R.string.chat_status_error_stalled));
            return;
        }
        mConsecutiveLlmErrors++;
    }

    /**
     * Capture via the WindowManager API, falling back to the mirror
     * VirtualDisplay + ImageReader if the primary path fails.
     */
    private Bitmap captureScreen() {
        // Primary: platform capture API (requires READ_FRAME_BUFFER)
        try {
            Bitmap screenshot = mScreenCaptureService.captureFrame();
            if (screenshot != null) {
                return screenshot;
            }
        } catch (Exception e) {
            Log.w(TAG, "WindowManager capture failed, trying fallback", e);
        }

        // Fallback: mirrored VirtualDisplay, created lazily on first failure
        if (mImageReader == null) {
            setupVirtualDisplay();
        }
        if (mImageReader != null) {
            try {
                Image image = mImageReader.acquireLatestImage();
                if (image != null) {
                    try {
                        return imageToBitmap(image);
                    } finally {
                        image.close();
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "ImageReader capture failed", e);
            }
        }
        return null;
    }

    /**
     * Create a mirror VirtualDisplay feeding an ImageReader. Requires
     * CAPTURE_VIDEO_OUTPUT (granted via the privapp-permissions XML).
     */
    private void setupVirtualDisplay() {
        try {
            DisplayManager displayManager = getSystemService(DisplayManager.class);
            Display display = displayManager != null
                    ? displayManager.getDisplay(Display.DEFAULT_DISPLAY) : null;
            if (display == null) {
                return;
            }
            android.graphics.Point size = new android.graphics.Point();
            display.getRealSize(size);
            android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
            display.getMetrics(metrics);

            mImageReader = ImageReader.newInstance(size.x, size.y, PixelFormat.RGBA_8888, 2);
            mVirtualDisplay = displayManager.createVirtualDisplay(
                    "AIPhoneAssistantMirror", size.x, size.y, metrics.densityDpi,
                    mImageReader.getSurface(),
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR);
            if (mVirtualDisplay == null) {
                Log.w(TAG, "Failed to create mirror VirtualDisplay");
                mImageReader.close();
                mImageReader = null;
            } else {
                Log.i(TAG, "Mirror VirtualDisplay created as capture fallback");
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Not allowed to create mirror display - check CAPTURE_VIDEO_OUTPUT", e);
            mImageReader = null;
        } catch (Exception e) {
            Log.e(TAG, "Error setting up VirtualDisplay", e);
            mImageReader = null;
        }
    }

    private Bitmap imageToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();

        Bitmap bitmap = Bitmap.createBitmap(
                image.getWidth() + rowPadding / pixelStride,
                image.getHeight(),
                Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        return bitmap;
    }

    private void releaseScreenCapture() {
        if (mVirtualDisplay != null) {
            mVirtualDisplay.release();
            mVirtualDisplay = null;
        }
        if (mImageReader != null) {
            mImageReader.close();
            mImageReader = null;
        }
    }

    // ------------------------------------------------------------------
    // Pause / interrupt / end paths
    // ------------------------------------------------------------------

    /** Interrupt bar listener — invoked on the UI thread. */
    private final InterruptBarView.Listener mInterruptBarListener =
            new InterruptBarView.Listener() {
        @Override
        public void onInterruptGestureStarted() {
            // Progress toasts below cover the hint; nothing extra needed here.
        }

        @Override
        public void onInterruptGestureProgress(int tapsRemaining) {
            Toast.makeText(AIPhoneAssistantService.this,
                    getString(R.string.toast_tap_n_more_format, tapsRemaining),
                    Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onInterruptRequested() {
            pauseForConfirmation();
        }
    };

    /** Pause the loop and ask the user whether to stop the task. */
    private void pauseForConfirmation() {
        if (!mIsRunning.get() || mPaused.get()) {
            return;
        }
        mPaused.set(true);
        ChatStore.setTaskState(this, ChatStore.STATE_PAUSED);
        mChatStore.updateConversation(mConvId, ChatStore.STATE_PAUSED, null);
        if (mOverlayController != null) {
            mOverlayController.hide();
        }
        updateFgNotification(getString(R.string.notification_paused));
        broadcastState(ChatStore.STATE_PAUSED);
        mOverlayController.showConfirmDialog(mCurrentTask, mDialogListener);
    }

    private final ConfirmDialogView.Listener mDialogListener = new ConfirmDialogView.Listener() {
        @Override
        public void onContinue() {
            mPaused.set(false);
            ChatStore.setTaskState(AIPhoneAssistantService.this, ChatStore.STATE_EXECUTING);
            mChatStore.updateConversation(mConvId, ChatStore.STATE_EXECUTING, null);
            mOverlayController.hideConfirmDialog();
            mOverlayController.showInterruptBar(mInterruptBarListener);
            // The user may have interacted while paused; re-baseline and give
            // the model a fresh look at whatever is on screen now.
            mFrameDetector.reset();
            mLastEscalationAtMs = System.currentTimeMillis();
            mForcedReanalyses = 0;
            mForcedRound = false;
            updateFgNotification(getString(R.string.notification_observing));
            broadcastState(ChatStore.STATE_EXECUTING);
        }

        @Override
        public void onStopConfirmed() {
            beginInterrupt();
        }
    };

    /** User confirmed the stop: wind down everything and finish. */
    private void beginInterrupt() {
        if (!mIsRunning.get()) {
            return;
        }
        mInterrupted = true;
        mPaused.set(false);
        mInputInjectionService.setAborted(true);
        mLLMClientService.cancelInFlight();
        mOverlayController.hideConfirmDialog();
        mOverlayController.hide();
        mCaptureHandler.post(this::endExecutionInterrupted);
    }

    private void endExecutionDone(String summary) {
        mChatStore.appendResult(mConvId,
                getString(R.string.chat_result_prefix_format, summary), mCurrentFrame);
        finishExecution(ChatStore.STATE_DONE,
                getString(R.string.notification_result_done_title),
                summary);
    }

    private void endExecutionInterrupted() {
        String text = getString(R.string.chat_status_interrupted);
        mChatStore.appendStatus(mConvId, text);
        finishExecution(ChatStore.STATE_INTERRUPTED,
                getString(R.string.notification_result_interrupted_title),
                text);
    }

    private void endExecutionStalled(String reason) {
        String text = reason != null ? reason : getString(R.string.chat_status_stalled);
        mChatStore.appendStatus(mConvId, text);
        finishExecution(ChatStore.STATE_STALLED,
                getString(R.string.notification_result_stalled_title),
                text);
    }

    /** Common teardown for every end path: notify, deactivate, stop. */
    private void finishExecution(String state, String notifTitle, String notifText) {
        mPaused.set(false);
        mInterrupted = true;
        ChatStore.setTaskState(this, state);
        mChatStore.updateConversation(mConvId, state, notifText);
        postResultNotification(notifTitle, notifText, mConvId);

        if (mOverlayController != null) {
            mOverlayController.hideConfirmDialog();
            mOverlayController.hideInterruptBar();
            mOverlayController.hide();
        }
        mCurrentFrame = null;
        mActionHistory.clear();

        broadcastState(state);
        mIsRunning.set(false);
        if (mCaptureHandler != null) {
            mCaptureHandler.removeCallbacks(mCaptureRunnable);
        }
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }

    // ------------------------------------------------------------------
    // Notifications
    // ------------------------------------------------------------------

    private void createNotificationChannels() {
        NotificationManager nm = getSystemService(NotificationManager.class);

        NotificationChannel channel = new NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_MIN);
        channel.setDescription(getString(R.string.notification_channel_description));
        channel.setShowBadge(false);
        nm.createNotificationChannel(channel);

        NotificationChannel result = new NotificationChannel(
                RESULT_CHANNEL_ID,
                getString(R.string.notification_channel_result_name),
                NotificationManager.IMPORTANCE_DEFAULT);
        result.setDescription(getString(R.string.notification_channel_result_description));
        nm.createNotificationChannel(result);
    }

    private PendingIntent chatContentIntent() {
        if (mConvId > 0 && mChatStore.getConversation(mConvId) != null) {
            return PendingIntent.getActivity(this, 0,
                    ChatActivity.intent(this, mConvId),
                    PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        }
        return PendingIntent.getActivity(this, 0,
                new Intent(this, ConversationListActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
    }

    private Notification buildFgNotification(String statusText) {
        String title = mCurrentTask != null && !mCurrentTask.isEmpty()
                ? getString(R.string.notification_task_title_format, shorten(mCurrentTask))
                : getString(R.string.notification_title);
        return new Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
                .setContentTitle(title)
                .setContentText(statusText)
                .setSmallIcon(android.R.drawable.ic_menu_info_details)
                .setOngoing(true)
                .setContentIntent(chatContentIntent())
                .build();
    }

    private void updateFgNotification(String statusText) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.notify(NOTIFICATION_ID, buildFgNotification(statusText));
    }

    private void postResultNotification(String title, String text, long convId) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        PendingIntent content = convId > 0
                ? PendingIntent.getActivity(this, 0,
                        ChatActivity.intent(this, convId),
                        PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE)
                : PendingIntent.getActivity(this, 0,
                        new Intent(this, ConversationListActivity.class),
                        PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, RESULT_CHANNEL_ID)
                .setContentTitle(title)
                .setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text))
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setAutoCancel(true)
                .setContentIntent(content)
                .build();
        nm.notify(RESULT_NOTIFICATION_ID, notification);
    }

    private void broadcastState(String state) {
        Intent intent = new Intent(BROADCAST_STATE_CHANGED);
        intent.putExtra(EXTRA_STATE, state);
        intent.putExtra(EXTRA_TASK, mCurrentTask != null ? mCurrentTask : "");
        intent.putExtra(EXTRA_CONV, mConvId);
        sendBroadcast(intent);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /** Human-readable one-liner for a model action (chat + notification). */
    private String describeAction(InputInjectionService.InputAction action) {
        switch (action.type) {
            case TAP:
                return getString(R.string.chat_action_tap_format,
                        (int) action.x, (int) action.y);
            case LONG_PRESS:
                return getString(R.string.chat_action_long_press_format,
                        (int) action.x, (int) action.y);
            case SWIPE:
                return getString(R.string.chat_action_swipe_format,
                        (int) action.x, (int) action.y,
                        (int) action.endX, (int) action.endY);
            case SCROLL:
                return getString(R.string.chat_action_scroll_format,
                        (int) action.x, (int) action.y);
            case KEY_PRESS:
                String keyName = KeyEvent.keyCodeToString(action.keyCode)
                        .replace("KEYCODE_", "");
                return getString(R.string.chat_action_key_format, keyName);
            case TEXT_INPUT:
                return getString(R.string.chat_action_text_format, action.text);
            case WAIT:
                return getString(R.string.chat_action_wait_format, action.durationMs);
            case OPEN_APP:
                return getString(R.string.chat_action_open_app_format,
                        action.app == null ? "" : action.app);
            case OPEN_SHORTCUT:
                return "open_shortcut " + (action.app == null ? "" : action.app)
                        + " / " + (action.shortcut == null ? "" : action.shortcut);
            case LIST_APPS:
                return getString(R.string.chat_action_list_apps_format,
                        action.query == null || action.query.isEmpty() ? "*" : action.query);
            default:
                return action.type.name();
        }
    }

    private static String shorten(String s) {
        if (s == null) {
            return "";
        }
        String flat = s.replace('\n', ' ').trim();
        return flat.length() <= TASK_TITLE_MAX_LEN
                ? flat : flat.substring(0, TASK_TITLE_MAX_LEN) + "…";
    }

    private static void sleepQuietly(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Binder for local service access. */
    public class AIPhoneAssistantBinder extends Binder {
        public AIPhoneAssistantService getService() {
            return AIPhoneAssistantService.this;
        }
    }
}
