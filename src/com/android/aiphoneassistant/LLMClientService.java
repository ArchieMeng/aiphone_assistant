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

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * LLM client for OpenAI-compatible chat completion APIs with vision support
 * (OpenAI, GLM-4V, Qwen-VL, vLLM, Ollama's /v1 endpoint, one-api gateways...).
 *
 * Configuration is read from the shared preferences written by SettingsActivity
 * on every request, so changes take effect without restarting the service:
 *   backend / endpoint / model / api_key / local_model_path / local_accel
 *
 * With backend "local" (only in builds that include LiteRT-LM, see
 * {@link LocalInference}) the same prompt and screenshot are run on-device,
 * e.g. by Gemma 4 E2B, instead of over HTTP; prompt building and reply
 * parsing are shared by both transports.
 *
 * The assistant is task-driven: the user's task text is included in the prompt
 * and the model is explicitly allowed to answer "none" when no action helps,
 * instead of inventing taps. Returned coordinates are in the coordinate space
 * of the (downscaled) image the model saw; this class rescales them back to
 * real screen pixels before handing the action to the caller.
 */
public class LLMClientService {
    private static final String TAG = "AIPhoneAssistant.LLM";

    public static final String PREFS_NAME = "ai_phone_assistant";
    public static final String KEY_ENABLED = "enabled";
    public static final String KEY_ENDPOINT = "endpoint";
    public static final String KEY_MODEL = "model";
    public static final String KEY_API_KEY = "api_key";
    public static final String KEY_TASK = "task";
    public static final String KEY_CAPTURE_INTERVAL = "capture_interval";
    public static final String KEY_BACKEND = "backend";
    public static final String KEY_LOCAL_MODEL_PATH = "local_model_path";
    public static final String KEY_LOCAL_ACCEL = "local_accel";
    // Fast/strong cascade (remote API): the fast model answers routine steps;
    // the strong model is consulted only when the fast loop is stuck. Empty
    // fast model falls back to KEY_MODEL; empty strong model disables escalation.
    public static final String KEY_MODEL_FAST = "model_fast";
    public static final String KEY_MODEL_STRONG = "model_strong";
    // Which transport serves escalated (strong) rounds: "" = same as the fast
    // backend, "remote" = OpenAI-compatible endpoint, "local" = on-device.
    // This allows fast = on-device and strong = online (or vice versa).
    public static final String KEY_STRONG_BACKEND = "strong_backend";
    // Strong rounds are allowed to think by default (they are the "ask the big
    // model to reason about the stuck screen" path). Can be turned off.
    public static final String KEY_STRONG_THINKING = "strong_thinking";
    public static final boolean DEFAULT_STRONG_THINKING = true;
    // Ask the provider to skip hidden chain-of-thought on the fast path. Only
    // fields documented by the matching API family are sent (see
    // applyFastResponseParams), so strict servers are not confused.
    public static final String KEY_DISABLE_THINKING = "disable_thinking";
    public static final boolean DEFAULT_DISABLE_THINKING = true;
    // Output cap: one action JSON is < 200 tokens. A low cap alone does not
    // stop a thinking model (reasoning tokens are billed against it), but it
    // bounds the worst case together with the disable-thinking hints.
    public static final String KEY_MAX_TOKENS = "max_tokens";
    public static final int DEFAULT_MAX_TOKENS = 1024;
    public static final String BACKEND_REMOTE = "remote";
    public static final String BACKEND_LOCAL = "local";
    public static final int DEFAULT_CAPTURE_INTERVAL_MS = 500;

    // Image compression settings. The model receives a downscaled image; the
    // scale factor between it and the real screen is applied to returned
    // coordinates so injections land where the model intended.
    private static final int JPEG_QUALITY = 80;
    private static final int MAX_IMAGE_SIZE = 1024;
    // On-device the vision encoder resizes anyway; a smaller JPEG keeps the
    // decode + preprocessing cheap.
    private static final int LOCAL_MAX_IMAGE_SIZE = 768;
    private static final int NORMALIZED_COORDS = 1000;

    // JSON schema for constrained decoding on-device: one action object with
    // the fields parseDecision understands, coordinates in 0-1000.
    private static final String COORD =
            "{\"type\":\"integer\",\"minimum\":0,\"maximum\":1000}";
    private static final String ACTION_SCHEMA = "{\"type\":\"object\",\"properties\":{"
            + "\"action\":{\"type\":\"string\",\"enum\":[\"tap\",\"swipe\","
            + "\"long_press\",\"scroll\",\"key_press\",\"text_input\",\"wait\","
            + "\"none\",\"done\"]},"
            + "\"x\":" + COORD + ",\"y\":" + COORD + ","
            + "\"endX\":" + COORD + ",\"endY\":" + COORD + ","
            + "\"deltaY\":{\"type\":\"integer\"},\"durationMs\":{\"type\":\"integer\"},"
            + "\"key\":{\"type\":\"string\"},\"text\":{\"type\":\"string\"},"
            + "\"summary\":{\"type\":\"string\"}},"
            + "\"required\":[\"action\"],\"additionalProperties\":false}";

    // HTTP timeouts
    private static final int CONNECT_TIMEOUT_MS = 10000;
    private static final int READ_TIMEOUT_MS = 60000;

    private final Context mContext;
    private final ExecutorService mExecutor;
    private final Handler mMainHandler;

    // The connection of the request currently in flight, kept so the task can
    // be interrupted (disconnect() aborts the blocking read on the executor).
    private final AtomicReference<HttpURLConnection> mInFlight = new AtomicReference<>();
    private final AtomicBoolean mCancelled = new AtomicBoolean(false);
    /** Null when this build has no on-device backend. */
    private final LocalInference mLocal;

    public LLMClientService(Context context) {
        mContext = context.getApplicationContext();
        mExecutor = Executors.newSingleThreadExecutor();
        mMainHandler = new Handler(Looper.getMainLooper());
        mLocal = LocalInference.get(mContext);
    }

    /** Read a config value from the shared preferences. */
    public static SharedPreferences getPrefs(Context context) {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    /** Local backend selected AND compiled into this build. */
    public static boolean isLocalBackend(SharedPreferences prefs) {
        return LocalInference.isAvailable()
                && BACKEND_LOCAL.equals(prefs.getString(KEY_BACKEND, BACKEND_REMOTE));
    }

    /** Local build + the strong-backend preference resolved to on-device. */
    public static boolean isLocalStrongBackend(SharedPreferences prefs) {
        String strongBackend = prefs.getString(KEY_STRONG_BACKEND, "").trim();
        if (strongBackend.isEmpty()) {
            return isLocalBackend(prefs);
        }
        return BACKEND_LOCAL.equals(strongBackend) && LocalInference.isAvailable();
    }

    /** Configured on-device model path, defaulting to the app's files dir. */
    public static String localModelPath(Context context) {
        String path = getPrefs(context).getString(KEY_LOCAL_MODEL_PATH, "").trim();
        return path.isEmpty() ? LocalInference.defaultModelFile(context).getPath() : path;
    }

    public static String localAccel(Context context) {
        return getPrefs(context).getString(KEY_LOCAL_ACCEL, LocalInference.ACCEL_GPU);
    }

    /** Whether the selected backend has what it needs to serve a request. */
    public static boolean isConfigured(Context context) {
        SharedPreferences prefs = getPrefs(context);
        if (isLocalBackend(prefs)) {
            return new java.io.File(localModelPath(context)).canRead();
        }
        return !prefs.getString(KEY_ENDPOINT, "").trim().isEmpty();
    }

    /**
     * Analyze a full-resolution screenshot against the given task.
     *
     * @param screenshot full-resolution screen bitmap
     * @param screenW real screen width in pixels
     * @param screenH real screen height in pixels
     * @param task the active task text (the assistant now has one task at a
     *             time, managed via ChatStore — no longer read from prefs)
     * @param recentActions descriptions of actions already performed for this
     *                      task, oldest first; included in the prompt so the
     *                      model avoids repeating itself and can tell when
     *                      the goal is reached
     * @param callback invoked on the main thread with a decision whose action
     *                 coordinates are already in real screen pixels
     */
    public void analyzeScreen(Bitmap screenshot, int screenW, int screenH, String task,
            List<String> recentActions, LLMCallback callback) {
        analyzeScreen(screenshot, screenW, screenH, task, recentActions, false, callback);
    }

    /**
     * Like the 6-argument overload, but with a fast/strong cascade hint: when
     * {@code preferStrong} is true and a strong model is configured, that model
     * serves this round. The caller escalates after repeated "none" answers,
     * request failures or a stale-screen watchdog round.
     */
    public void analyzeScreen(Bitmap screenshot, int screenW, int screenH, String task,
            List<String> recentActions, boolean preferStrong, LLMCallback callback) {
        SharedPreferences prefs = getPrefs(mContext);
        String endpoint = prefs.getString(KEY_ENDPOINT, "").trim();
        String apiKey = prefs.getString(KEY_API_KEY, "").trim();
        String legacyModel = prefs.getString(KEY_MODEL, "gpt-4o").trim();
        String fastModel = prefs.getString(KEY_MODEL_FAST, "").trim();
        if (fastModel.isEmpty()) {
            fastModel = legacyModel;
        }
        String strongModel = prefs.getString(KEY_MODEL_STRONG, "").trim();
        boolean fastLocal = isLocalBackend(prefs);
        boolean strongLocal = isLocalStrongBackend(prefs);
        // A strong round is possible if its transport is configured: for the
        // remote transport that needs a model name; for the local transport it
        // just needs the LiteRT-LM backend to be compiled in.
        boolean strongConfigured = strongLocal || !strongModel.isEmpty();
        boolean escalated = preferStrong && strongConfigured;
        boolean local = escalated ? strongLocal : fastLocal;
        if (escalated && !local && endpoint.isEmpty()) {
            // Strong backend is remote but no endpoint: keep the fast backend.
            Log.w(TAG, "Escalation requested but the remote endpoint is empty; "
                    + "staying on the fast backend");
            escalated = false;
            local = fastLocal;
        }
        String model = escalated && !strongLocal ? strongModel : fastModel;
        boolean disableThinking = prefs.getBoolean(KEY_DISABLE_THINKING,
                DEFAULT_DISABLE_THINKING);
        String modelPath = localModelPath(mContext);
        String accel = localAccel(mContext);

        if ((!local && endpoint.isEmpty()) || task == null || task.trim().isEmpty()) {
            if (callback != null) {
                callback.onError("LLM endpoint or task not configured");
            }
            return;
        }

        mCancelled.set(false);
        // Resolve the per-round choices into effectively-final values so the
        // request thread can capture them (the method reassigns local/escalated
        // while validating the configuration above).
        final boolean localBackend = local;
        final boolean escalatedRound = escalated;
        final boolean allowThinking = escalatedRound && prefs.getBoolean(
                KEY_STRONG_THINKING, DEFAULT_STRONG_THINKING);
        final boolean thinkingSuppressed = disableThinking && !allowThinking;
        // A dedicated thread per request: a hung gateway response must never
        // block later analysis rounds (the single-thread executor would).
        Thread t = new Thread(() -> {
            try {
                Bitmap scaled = scaleBitmap(screenshot,
                        localBackend ? LOCAL_MAX_IMAGE_SIZE : MAX_IMAGE_SIZE);
                // Gemma grounds in Gemini-style coordinates normalized to
                // 0-1000 regardless of the image size (asking for pixels
                // still yields 0-1000 values), so the local prompt asks for
                // that space and replies are rescaled from 1000x1000.
                int coordW = localBackend ? NORMALIZED_COORDS : scaled.getWidth();
                int coordH = localBackend ? NORMALIZED_COORDS : scaled.getHeight();
                // Static instructions are the cacheable system message; the
                // per-round task/history are the user message.
                String instructions = buildInstructions(localBackend, escalatedRound);
                String prompt = buildUserPrompt(task, scaled.getWidth(),
                        scaled.getHeight(), recentActions);
                String response;
                if (localBackend) {
                    requireLocal();
                    response = mLocal.generate(modelPath, accel,
                            instructions + "\n\n" + prompt, bitmapToJpeg(scaled),
                            ACTION_SCHEMA);
                    Log.i(TAG, "Local reply: " + shorten(response, 200));
                } else {
                    Log.i(TAG, "Round model: " + model
                            + (escalatedRound ? " (strong, escalated)" : " (fast)")
                            + ", backend=" + (localBackend ? "local" : "remote")
                            + ", thinking=" + (thinkingSuppressed ? "off" : "on")
                            + ", prompt chars=" + (instructions.length() + prompt.length()));
                    String imageBase64 = Base64.encodeToString(bitmapToJpeg(scaled),
                            Base64.NO_WRAP);
                    response = sendToLLM(endpoint, apiKey, model, instructions, prompt,
                            imageBase64, disableThinking, allowThinking);
                }
                LLMDecision parsed = parseDecision(response, screenW, screenH,
                        coordW, coordH);
                // The prompt history must be in the model's coordinate space,
                // or it copies screen-pixel values back as 0-1000 coordinates.
                // The constrained reply is exactly the action JSON.
                LLMDecision decision = localBackend
                        ? parsed.withHistoryEntry(shorten(response, 200)) : parsed;
                if (scaled != screenshot) {
                    scaled.recycle();
                }
                mMainHandler.post(() -> {
                    if (callback != null) {
                        callback.onDecision(decision);
                    }
                });
            } catch (Exception e) {
                if (mCancelled.get()) {
                    Log.i(TAG, "LLM request cancelled");
                    mMainHandler.post(() -> {
                        if (callback != null) {
                            callback.onInterrupted();
                        }
                    });
                    return;
                }
                Log.e(TAG, "LLM analysis failed", e);
                mMainHandler.post(() -> {
                    if (callback != null) {
                        callback.onError(e.getMessage());
                    }
                });
            }
        });
        t.setName("llm-analyze");
        t.start();
    }

    /**
     * Abort the request currently in flight (used by the user interrupt). The
     * blocked read fails on the executor thread and surfaces as
     * {@link LLMCallback#onInterrupted()}.
     */
    public void cancelInFlight() {
        mCancelled.set(true);
        if (mLocal != null) {
            mLocal.cancel();
        }
        HttpURLConnection connection = mInFlight.get();
        if (connection != null) {
            connection.disconnect();
        }
    }

    /**
     * Send a text-only ping to the configured endpoint; used by Settings'
     * "Test Connection" button. Callback is invoked on the main thread.
     */
    public void testConnection(TestCallback callback) {
        SharedPreferences prefs = getPrefs(mContext);
        String endpoint = prefs.getString(KEY_ENDPOINT, "").trim();
        String apiKey = prefs.getString(KEY_API_KEY, "").trim();
        String model = prefs.getString(KEY_MODEL, "gpt-4o").trim();

        if (isLocalBackend(prefs)) {
            testLocal(callback);
            return;
        }
        if (endpoint.isEmpty()) {
            if (callback != null) {
                callback.onResult(false, "Endpoint not configured");
            }
            return;
        }

        mExecutor.execute(() -> {
            try {
                String reply = sendToLLM(endpoint, apiKey, model,
                        "You are a connectivity probe.",
                        "Reply with the single word OK and nothing else.", null, false, false);
                mMainHandler.post(() -> {
                    if (callback != null) {
                        callback.onResult(true, "Model replied: " + shorten(reply, 60));
                    }
                });
            } catch (Exception e) {
                String message = e.getMessage() != null ? e.getMessage() : e.toString();
                mMainHandler.post(() -> {
                    if (callback != null) {
                        callback.onResult(false, shorten(message, 200));
                    }
                });
            }
        });
    }

    /** Load the on-device model (if needed) and run a tiny text prompt. */
    private void testLocal(TestCallback callback) {
        String modelPath = localModelPath(mContext);
        String accel = localAccel(mContext);
        mExecutor.execute(() -> {
            try {
                long t0 = android.os.SystemClock.elapsedRealtime();
                requireLocal();
                String reply = mLocal.generate(modelPath, accel,
                        "Reply with the single word OK and nothing else.", null, null);
                long ms = android.os.SystemClock.elapsedRealtime() - t0;
                String detail = "Model replied: " + shorten(reply, 40) + " (" + ms + " ms, "
                        + mLocal.loadedAccel() + ")";
                mMainHandler.post(() -> {
                    if (callback != null) {
                        callback.onResult(true, detail);
                    }
                });
            } catch (Exception e) {
                Log.e(TAG, "Local model test failed", e);
                String message = e.getMessage() != null ? e.getMessage() : e.toString();
                mMainHandler.post(() -> {
                    if (callback != null) {
                        callback.onResult(false, shorten(message, 200));
                    }
                });
            }
        });
    }

    private void requireLocal() throws IOException {
        if (mLocal == null) {
            throw new IOException("On-device backend failed to initialize");
        }
    }

    private static String shorten(String s, int max) {
        if (s == null) {
            return "";
        }
        s = s.replace('\n', ' ').trim();
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }

    /**
     * Static, per-round-identical instructions. Sent as the system message so
     * OpenAI-compatible providers can reuse their prompt cache, and so the
     * fast model never has to re-read the action menu from scratch.
     *
     * The prompt is deliberately terse and explicitly forbids chain-of-thought:
     * a screen-action model that "thinks" for thousands of tokens turns every
     * 2-second step into a 10-30 second step for zero accuracy gain.
     */
    private String buildInstructions(boolean normalizedCoords, boolean strong) {
        StringBuilder p = new StringBuilder();
        if (strong) {
            // Escalation rounds may think: the fast policy is stuck, and a
            // plan/reasoning pass is exactly what the strong model is for.
            p.append("You are the STRONG escalation model for an Android phone agent. ")
                    .append("The fast policy has been stuck (it repeated screens/actions). ")
                    .append("You may reason about the screenshot and the action history ")
                    .append("before answering. Start a NEW strategy instead of repeating the ")
                    .append("failed actions (e.g. use open_app, BACK, scroll, or wait). ")
                    .append("Finish with exactly ONE JSON object from the list below, ")
                    .append("as the last thing in your reply, and nothing after it.\n");
        } else {
            p.append("You control an Android phone from ONE screenshot per step. ")
                    .append("Answer IMMEDIATELY with exactly one JSON object: no analysis, ")
                    .append("no step-by-step reasoning, no prose.\n");
        }
        p.append("Actions (choose one):\n")
                .append("{\"action\":\"tap\",\"x\":<int>,\"y\":<int>}\n")
                .append("{\"action\":\"swipe\",\"x\":<int>,\"y\":<int>,\"endX\":<int>,")
                .append("\"endY\":<int>,\"durationMs\":<int>}\n")
                .append("{\"action\":\"long_press\",\"x\":<int>,\"y\":<int>,")
                .append("\"durationMs\":<int>}\n")
                .append("{\"action\":\"scroll\",\"x\":<int>,\"y\":<int>,")
                .append("\"deltaY\":<int>}  // negative deltaY scrolls content up\n")
                .append("{\"action\":\"key_press\",\"key\":\"BACK\"}  // BACK|HOME|")
                .append("APP_SWITCH|ENTER|DEL|DPAD_UP|DPAD_DOWN|VOLUME_UP...\n")
                .append("{\"action\":\"text_input\",\"text\":\"...\"}  ")
                .append("// focused text field; ANY language/Unicode (pasted), ")
                .append("e.g. Chinese search queries\n")
                .append("{\"action\":\"open_app\",\"app\":\"<name or package>\",")
                .append("\"uri\":\"<optional>\"}  // cheapest way to switch apps\n")
                .append("{\"action\":\"open_shortcut\",\"app\":\"<name or package>\",")
                .append("\"shortcut\":\"<id or label from list_apps sc:...>\"}  ")
                .append("// jump straight to an app shortcut (e.g. Bilibili/YouTube search)\n")
                .append("{\"action\":\"double_tap\",\"x\":<int>,\"y\":<int>}\n")
                .append("{\"action\":\"open_uri\",\"uri\":\"<https://... or app deep link>\"}\n")
                .append("{\"action\":\"expand_notifications\"}  // pull down the notification shade\n")
                .append("{\"action\":\"expand_quick_settings\"}  // pull down quick settings\n")
                .append("{\"action\":\"list_apps\",\"query\":\"<filter>\"}  // only if the name is unknown\n")
                .append("{\"action\":\"wait\",\"durationMs\":<int>}  // screen still loading\n")
                .append("{\"action\":\"none\"}  // no useful action right now\n")
                .append("{\"action\":\"done\",\"summary\":\"<one short sentence>\"}  ")
                .append("// goal VISIBLY achieved; summary in the task's language\n")
                .append("BATCH: instead of one action you may return ")
                .append("{\"steps\":[{...},{...}]} with up to 12 gesture steps ")
                .append("(tap/swipe/drag/long_press/double_tap/key_press/text_input/scroll/wait). ")
                .append("They run in order and then a new screenshot is taken, which is much ")
                .append("faster for known blind sequences (search flow, game combos, repeated ")
                .append("taps). Do NOT batch steps whose choice depends on intermediate screens, ")
                .append("and never put open_app/open_shortcut/open_uri/list_apps inside steps.\n")
                .append(normalizedCoords
                        ? "Coordinates are normalized to 0-" + NORMALIZED_COORDS
                                + " on both axes (0,0 top-left). "
                        : "Coordinates are image pixels. ")
                .append("Rules: tap only visible controls; never repeat a failed action; ")
                .append("prefer open_app over hunting for icons; wait while loading; ")
                .append("done only when the goal is clearly visible on this screen. ")
                .append("To search inside an app: if list_apps shows an sc:search shortcut, use ")
                .append("open_shortcut; otherwise open_app -> tap its search box -> ")
                .append("text_input with the search words -> key_press ENTER -> tap a result.");
        return p.toString();
    }

    /**
     * Per-round user message: the task plus the bounded action history. Keeping
     * this short matters more than the instructions: it changes every round and
     * therefore cannot be served from the provider's prompt cache.
     */
    private String buildUserPrompt(String task, int imageWidth, int imageHeight,
            List<String> recentActions) {
        StringBuilder p = new StringBuilder();
        p.append("Screenshot: ").append(imageWidth).append("x").append(imageHeight)
                .append(" px. Task: \"").append(task).append("\"\n");
        if (recentActions != null && !recentActions.isEmpty()) {
            p.append("Done so far (oldest first):\n");
            for (int i = 0; i < recentActions.size(); i++) {
                p.append(i + 1).append(". ").append(recentActions.get(i)).append('\n');
            }
            p.append("Do not repeat an action that did not help.\n");
        }
        p.append("Next action JSON:");
        return p.toString();
    }

    /** Downscale so the largest dimension is at most {@code maxSize}. */
    private Bitmap scaleBitmap(Bitmap bitmap, int maxSize) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int maxDim = Math.max(width, height);
        if (maxDim <= maxSize) {
            return bitmap;
        }
        float scale = (float) maxSize / maxDim;
        return Bitmap.createScaledBitmap(bitmap,
                Math.round(width * scale), Math.round(height * scale), true);
    }

    private byte[] bitmapToJpeg(Bitmap bitmap) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out);
        return out.toByteArray();
    }

    /**
     * Ask the provider to skip hidden chain-of-thought on the fast path.
     * Only fields documented by the matching API family are added: OpenAI-style
     * servers reject unknown top-level parameters with HTTP 400, so guessing a
     * generic field would break more deployments than it speeds up.
     */
    private void applyFastResponseParams(JSONObject body, String endpoint, String model,
            boolean disableThinking) throws JSONException {
        if (!disableThinking) {
            return;
        }
        String e = endpoint == null ? "" : endpoint.toLowerCase(java.util.Locale.ROOT);
        String m = model == null ? "" : model.toLowerCase(java.util.Locale.ROOT);
        if (e.contains("openrouter.ai")) {
            // https://openrouter.ai/docs/features/reasoning
            JSONObject reasoning = new JSONObject();
            reasoning.put("enabled", false);
            body.put("reasoning", reasoning);
            return;
        }
        if (e.contains("dashscope.aliyuncs.com") || e.contains("aliyuncs.com")) {
            // Qwen / DashScope OpenAI-compatible mode.
            JSONObject template = new JSONObject();
            template.put("enable_thinking", false);
            body.put("chat_template_kwargs", template);
            body.put("enable_thinking", false);
            return;
        }
        if (e.contains("api.openai.com")
                && (m.startsWith("o1") || m.startsWith("o3") || m.startsWith("o4")
                        || m.startsWith("gpt-5"))) {
            body.put("reasoning_effort", "minimal");
            return;
        }
        if (e.contains("deepseek.com") && (m.contains("reasoner") || m.contains("thinking"))) {
            // DeepSeek does not document a request-level switch in this shape;
            // the supported way to get instant replies is a non-thinking model
            // name (e.g. deepseek-chat) as the fast model.
            Log.w(TAG, "Thinking model '" + model + "' is the fast model; prefer a "
                    + "non-thinking model (e.g. deepseek-chat) for instant replies");
        }
    }

    /**
     * POST a chat completion request. When imageBase64 is null the user message
     * is sent as plain text (used by the connection test).
     */
    private String sendToLLM(String endpoint, String apiKey, String model, String instructions,
            String userPrompt, String imageBase64, boolean disableThinking,
            boolean allowThinking) throws IOException {
        JSONObject requestBody = new JSONObject();
        // Strong rounds are allowed to think, so they get a much larger output
        // budget; fast rounds keep the tight cap.
        int maxTokens = DEFAULT_MAX_TOKENS;
        try {
            int configured = getPrefs(mContext).getInt(KEY_MAX_TOKENS, DEFAULT_MAX_TOKENS);
            if (configured > 0) {
                maxTokens = configured;
            }
        } catch (Exception ignored) {
            // keep the default
        }
        if (allowThinking) {
            maxTokens = Math.max(4096, maxTokens);
        }
        try {
            requestBody.put("model", model);

            JSONArray messages = new JSONArray();
            // Stable prefix first: providers with prompt caching (DeepSeek,
            // OpenAI, Qwen) bill and prefill it much faster on repeat rounds.
            JSONObject systemMessage = new JSONObject();
            systemMessage.put("role", "system");
            systemMessage.put("content", instructions);
            messages.put(systemMessage);
            JSONObject userMessage = new JSONObject();
            userMessage.put("role", "user");
            if (imageBase64 != null) {
                JSONArray content = new JSONArray();
                JSONObject textContent = new JSONObject();
                textContent.put("type", "text");
                textContent.put("text", userPrompt);
                content.put(textContent);
                JSONObject imageContent = new JSONObject();
                imageContent.put("type", "image_url");
                JSONObject imageUrl = new JSONObject();
                imageUrl.put("url", "data:image/jpeg;base64," + imageBase64);
                imageContent.put("image_url", imageUrl);
                content.put(imageContent);
                userMessage.put("content", content);
            } else {
                userMessage.put("content", userPrompt);
            }
            messages.put(userMessage);
            requestBody.put("messages", messages);
            // One action JSON is far below this cap; the cap only bounds a
            // runaway reply. Configurable via KEY_MAX_TOKENS.
            requestBody.put("max_tokens", maxTokens);
            // Stream (SSE): a non-streaming POST sits with zero bytes and can
            // be cut by intermediate proxies.
            requestBody.put("stream", true);
            // Thinking is only suppressed on the fast path. Strong rounds are
            // the "call the big model and let it reason" path by design.
            applyFastResponseParams(requestBody, endpoint, model, disableThinking && !allowThinking);
        } catch (JSONException e) {
            throw new IOException("Failed to build request", e);
        }

        HttpURLConnection connection = (HttpURLConnection) new URL(endpoint).openConnection();
        mInFlight.set(connection);
        try {
            long t0 = android.os.SystemClock.elapsedRealtime();
            java.net.InetAddress[] addrs = java.net.InetAddress.getAllByName(
                    new URL(endpoint).getHost());
            Log.i(TAG, "DNS: " + new URL(endpoint).getHost() + " -> "
                    + java.util.Arrays.toString(addrs));
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "application/json");
            if (!apiKey.isEmpty()) {
                connection.setRequestProperty("Authorization", "Bearer " + apiKey);
            }

            byte[] payload = requestBody.toString().getBytes(StandardCharsets.UTF_8);
            try (OutputStream os = connection.getOutputStream()) {
                os.write(payload);
            }

            int responseCode = connection.getResponseCode();
            Log.i(TAG, "HTTP response " + responseCode + " after "
                    + (android.os.SystemClock.elapsedRealtime() - t0) + " ms");
            if (responseCode != HttpURLConnection.HTTP_OK) {
                String error = readStream(connection.getErrorStream());
                throw new IOException("HTTP " + responseCode + " "
                        + shorten(error.isEmpty() ? connection.getResponseMessage() : error, 300));
            }

            String content = readStreamSSE(connection);
            if (content.trim().isEmpty()) {
                throw new IOException("Empty streamed response from model");
            }
            return content;
        } finally {
            mInFlight.compareAndSet(connection, null);
            connection.disconnect();
        }
    }

    /**
     * Consume an OpenAI-compatible SSE stream, returning the concatenated
     * answer (content, falling back to the reasoning field for thinking
     * models that never emit a final content delta).
     */
    private String readStreamSSE(HttpURLConnection connection) throws IOException {
        long t0 = android.os.SystemClock.elapsedRealtime();
        StringBuilder content = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        boolean gotAnything = false;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String data = line.substring(5).trim();
                if (!gotAnything) {
                    gotAnything = true;
                    Log.i(TAG, "First stream line after "
                            + (android.os.SystemClock.elapsedRealtime() - t0)
                            + " ms: " + line);
                }
                if (data.equals("[DONE]")) {
                    break;
                }
                try {
                    JSONObject chunk = new JSONObject(data);
                    JSONArray choices = chunk.optJSONArray("choices");
                    if (choices == null || choices.length() == 0) {
                        continue;
                    }
                    JSONObject delta = choices.getJSONObject(0).optJSONObject("delta");
                    if (delta == null) {
                        continue;
                    }
                    content.append(delta.optString("content", ""));
                    reasoning.append(delta.optString("reasoning_content", ""));
                } catch (JSONException ignored) {
                    // keep-alive comments / partial lines
                }
            }
        }
        String answer = content.toString().trim();
        if (answer.isEmpty()) {
            answer = reasoning.toString().trim();
            if (!answer.isEmpty()) {
                Log.i(TAG, "Empty content; falling back to reasoning_content");
            }
        }
        return answer;
    }

    private String readStream(InputStream inputStream) throws IOException {
        if (inputStream == null) {
            return "";
        }
        StringBuilder response = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
        }
        return response.toString();
    }

    /**
     * Parse the model's reply into a {@link LLMDecision} with REAL screen
     * coordinates. Malformed replies or actions with missing required fields
     * yield {@code none} — never a guessed default tap.
     */
    private LLMDecision parseDecision(String response, int screenW,
            int screenH, int imageW, int imageH) {
        if (response == null) {
            return LLMDecision.none(null);
        }
        try {
            int jsonStart = response.indexOf('{');
            int jsonEnd = response.lastIndexOf('}');
            if (jsonStart < 0 || jsonEnd <= jsonStart) {
                Log.w(TAG, "No JSON found in response: " + shorten(response, 200));
                return LLMDecision.none(response);
            }
            JSONObject json = new JSONObject(response.substring(jsonStart, jsonEnd + 1));
            String actionType = json.optString("action", "").trim().toLowerCase();

            // Task completion. The legacy synonyms map to done as well — models
            // trained on older prompts still answer finish/complete/stop.
            switch (actionType) {
                case "done":
                case "finish":
                case "complete":
                case "stop": {
                    String summary = json.optString("summary", "").trim();
                    if (summary.isEmpty()) {
                        summary = json.optString("text", "").trim();
                    }
                    Log.i(TAG, "Model reports task done: " + shorten(summary, 80));
                    return LLMDecision.done(summary, response);
                }
                case "none":
                    Log.i(TAG, "Model chose no action");
                    return LLMDecision.none(response);
                default:
                    break;
            }

            InputInjectionService.InputAction action;
            switch (actionType) {
                case "tap":
                case "click":
                    if (!json.has("x") || !json.has("y")) {
                        Log.w(TAG, "tap without coordinates, ignoring");
                        return LLMDecision.none(response);
                    }
                    action = InputInjectionService.InputAction.tap(
                            (float) json.getDouble("x"), (float) json.getDouble("y"));
                    break;

                case "swipe":
                    if (!json.has("x") || !json.has("y")
                            || !json.has("endX") || !json.has("endY")) {
                        Log.w(TAG, "swipe missing endpoints, ignoring");
                        return LLMDecision.none(response);
                    }
                    action = InputInjectionService.InputAction.swipe(
                            (float) json.getDouble("x"), (float) json.getDouble("y"),
                            (float) json.getDouble("endX"), (float) json.getDouble("endY"),
                            clampMs(json.optLong("durationMs", 300)));
                    break;

                case "long_press":
                    if (!json.has("x") || !json.has("y")) {
                        return LLMDecision.none(response);
                    }
                    action = InputInjectionService.InputAction.longPress(
                            (float) json.getDouble("x"), (float) json.getDouble("y"),
                            clampMs(json.optLong("durationMs", 800)));
                    break;

                case "scroll":
                case "scroll_up":
                case "scroll_down":
                    if (!json.has("x") || !json.has("y")) {
                        return LLMDecision.none(response);
                    }
                    float deltaY = (float) json.optDouble("deltaY",
                            "scroll_down".equals(actionType) ? 500 : -500);
                    action = InputInjectionService.InputAction.scroll(
                            (float) json.getDouble("x"), (float) json.getDouble("y"), deltaY);
                    break;

                case "key_press": {
                    int keyCode = 0;
                    if (json.has("key")) {
                        keyCode = InputInjectionService.keyNameToKeyCode(json.getString("key"));
                    } else if (json.has("keyCode")) {
                        keyCode = json.optInt("keyCode", 0);
                        if (keyCode == 0) {
                            keyCode = InputInjectionService.keyNameToKeyCode(
                                    json.optString("keyCode"));
                        }
                    }
                    if (keyCode == 0) {
                        Log.w(TAG, "key_press with unknown key: " + json);
                        return LLMDecision.none(response);
                    }
                    action = InputInjectionService.InputAction.keyPress(keyCode);
                    break;
                }

                case "text_input":
                case "type": {
                    String text = json.optString("text", "");
                    if (text.isEmpty()) {
                        return LLMDecision.none(response);
                    }
                    action = InputInjectionService.InputAction.textInput(text);
                    break;
                }

                case "wait":
                    action = InputInjectionService.InputAction.waitAction(
                            json.optLong("durationMs", 1000));
                    break;

                default:
                    Log.w(TAG, "Unknown action type: " + actionType);
                    return LLMDecision.none(response);
            }

            // The model saw the downscaled image; map coordinates back to the
            // real screen so injections land where the model intended.
            float sx = screenW / (float) imageW;
            float sy = screenH / (float) imageH;
            action.x *= sx;
            action.y *= sy;
            action.endX *= sx;
            action.endY *= sy;
            action.deltaY *= sy;
            return LLMDecision.action(action, response);
        } catch (Exception e) {
            Log.e(TAG, "Error parsing response", e);
            return LLMDecision.none(response);
        }
    }

    private static long clampMs(long value) {
        return Math.min(Math.max(value, 50), 10000);
    }

    public void release() {
        mExecutor.shutdown();
    }

    /**
     * Free the shared on-device model (it is otherwise kept resident between
     * requests). Called when the assistant service goes away.
     */
    public static void unloadLocalModel(Context context) {
        LocalInference local = LocalInference.get(context);
        if (local == null) {
            return;
        }
        // Closing waits for any in-flight generation; keep it off the caller.
        local.cancel();
        new Thread(local::close, "llm-local-close").start();
    }

    /** Callback for screen analysis. */
    public interface LLMCallback {
        void onDecision(LLMDecision decision);
        /** The in-flight request was cancelled by the user interrupt. */
        void onInterrupted();
        void onError(String error);
    }

    /** Callback for the connection test. */
    public interface TestCallback {
        void onResult(boolean success, String detail);
    }

    /**
     * What the model decided to do: perform an {@link #ACTION}, report the
     * task {@link #DONE} (with a summary) or {@link #NONE} (nothing useful
     * to do right now).
     */
    public static class LLMDecision {
        public enum Kind { ACTION, DONE, NONE }

        public final Kind kind;
        /** Non-null only for {@link Kind#ACTION}; coordinates in screen pixels. */
        public final InputInjectionService.InputAction action;
        /** One-sentence summary of what was achieved; only for {@link Kind#DONE}. */
        public final String summary;
        public final String raw;
        /**
         * How to describe the action in later prompts, in the model's own
         * coordinate space; null = use the screen-pixel description.
         */
        public final String historyEntry;

        private LLMDecision(Kind kind, InputInjectionService.InputAction action,
                String summary, String raw) {
            this(kind, action, summary, raw, null);
        }

        private LLMDecision(Kind kind, InputInjectionService.InputAction action,
                String summary, String raw, String historyEntry) {
            this.kind = kind;
            this.action = action;
            this.summary = summary;
            this.raw = raw;
            this.historyEntry = historyEntry;
        }

        LLMDecision withHistoryEntry(String entry) {
            return new LLMDecision(kind, action, summary, raw, entry);
        }

        public static LLMDecision action(InputInjectionService.InputAction action, String raw) {
            return new LLMDecision(Kind.ACTION, action, null, raw);
        }

        public static LLMDecision done(String summary, String raw) {
            return new LLMDecision(Kind.DONE, null, summary, raw);
        }

        public static LLMDecision none(String raw) {
            return new LLMDecision(Kind.NONE, null, null, raw);
        }
    }
}
