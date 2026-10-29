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

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.hardware.input.InputManager;
import android.os.SystemClock;
import android.util.Log;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.KeyCharacterMap;
import android.view.MotionEvent;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Input injection via the platform InputManager API.
 *
 * Built with platform_apis: true, so {@link InputManager#injectInputEvent} is
 * called directly (the framework enforces the INJECT_EVENTS permission, which
 * this app holds via the privapp-permissions XML).
 *
 * Note: all execute* methods block (they sleep between injected events), so
 * they must be called from a background thread, never the UI thread.
 */
public class InputInjectionService {
    private static final String TAG = "AIPhoneAssistant.Input";

    private final AIPhoneAssistantService mParentService;
    private InputManager mInputManager;

    // Key names accepted from LLM responses, e.g. {"action":"key_press","key":"BACK"}
    private static final Map<String, Integer> KEY_NAMES = buildKeyNames();

    private float mLastTouchX;
    private float mLastTouchY;
    private long mLastTouchTime;

    // Set by the service when the user interrupts the task: gestures in
    // progress are wound up as soon as possible (always still sending the
    // closing ACTION_UP so no touch stays stuck down).
    private volatile boolean mAborted = false;

    public InputInjectionService(AIPhoneAssistantService parent) {
        mParentService = parent;
        initialize();
    }

    /** Request that the gesture currently in progress is wound up. */
    public void setAborted(boolean aborted) {
        mAborted = aborted;
    }

    public boolean isAborted() {
        return mAborted;
    }

    private void initialize() {
        Log.i(TAG, "Initializing InputInjectionService");
        try {
            mInputManager = InputManager.getInstance();
            Log.i(TAG, "InputInjectionService initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize InputInjectionService", e);
        }
    }

    /**
     * Resolve a key name like "back", "BACK" or "KEYCODE_BACK" to a keycode.
     * Returns 0 when unknown.
     */
    public static int keyNameToKeyCode(String name) {
        if (name == null) {
            return 0;
        }
        String normalized = name.trim().toUpperCase(Locale.US);
        if (normalized.startsWith("KEYCODE_")) {
            normalized = normalized.substring("KEYCODE_".length());
        }
        Integer code = KEY_NAMES.get(normalized);
        return code != null ? code : 0;
    }

    private static Map<String, Integer> buildKeyNames() {
        Map<String, Integer> map = new HashMap<>();
        map.put("BACK", KeyEvent.KEYCODE_BACK);
        map.put("ESCAPE", KeyEvent.KEYCODE_BACK); // convenience alias
        map.put("HOME", KeyEvent.KEYCODE_HOME);
        map.put("APP_SWITCH", KeyEvent.KEYCODE_APP_SWITCH);
        map.put("RECENTS", KeyEvent.KEYCODE_APP_SWITCH);
        map.put("ENTER", KeyEvent.KEYCODE_ENTER);
        map.put("DPAD_CENTER", KeyEvent.KEYCODE_DPAD_CENTER);
        map.put("DEL", KeyEvent.KEYCODE_DEL);
        map.put("DELETE", KeyEvent.KEYCODE_DEL);
        map.put("BACKSPACE", KeyEvent.KEYCODE_DEL);
        map.put("SPACE", KeyEvent.KEYCODE_SPACE);
        map.put("TAB", KeyEvent.KEYCODE_TAB);
        map.put("DPAD_UP", KeyEvent.KEYCODE_DPAD_UP);
        map.put("DPAD_DOWN", KeyEvent.KEYCODE_DPAD_DOWN);
        map.put("DPAD_LEFT", KeyEvent.KEYCODE_DPAD_LEFT);
        map.put("DPAD_RIGHT", KeyEvent.KEYCODE_DPAD_RIGHT);
        map.put("VOLUME_UP", KeyEvent.KEYCODE_VOLUME_UP);
        map.put("VOLUME_DOWN", KeyEvent.KEYCODE_VOLUME_DOWN);
        map.put("VOLUME_MUTE", KeyEvent.KEYCODE_VOLUME_MUTE);
        map.put("MUTE", KeyEvent.KEYCODE_VOLUME_MUTE);
        map.put("POWER", KeyEvent.KEYCODE_POWER);
        map.put("MENU", KeyEvent.KEYCODE_MENU);
        map.put("NOTIFICATION", KeyEvent.KEYCODE_NOTIFICATION);
        map.put("SEARCH", KeyEvent.KEYCODE_SEARCH);
        map.put("CAMERA", KeyEvent.KEYCODE_CAMERA);
        return map;
    }

    /**
     * Execute an input action from the LLM response. NONE is a no-op, WAIT
     * simply sleeps so the screen can settle before the next analysis.
     * Returns false when the gesture was aborted mid-way by the user
     * interrupt (the gesture is always closed with a final ACTION_UP).
     */
    public boolean executeAction(InputAction action) {
        if (action == null || mAborted) {
            return false;
        }

        switch (action.type) {
            case TAP:
                tap(action.x, action.y);
                break;
            case SWIPE:
                return swipe(action.x, action.y, action.endX, action.endY, action.durationMs);
            case LONG_PRESS:
                longPress(action.x, action.y, action.durationMs);
                break;
            case KEY_PRESS:
                keyPress(action.keyCode);
                break;
            case TEXT_INPUT:
                textInput(action.text);
                break;
            case SCROLL:
                scroll(action.x, action.y, action.deltaY);
                break;
            case WAIT:
                Log.i(TAG, "Waiting " + action.durationMs + "ms for screen to settle");
                sleepCancellable(action.durationMs);
                break;
            case NONE:
            default:
                Log.d(TAG, "No-op action: " + action.type);
                break;
        }
        return !mAborted;
    }

    private void injectMotionEvent(int action, float x, float y, long downTime) {
        long now = SystemClock.uptimeMillis();
        MotionEvent event = MotionEvent.obtain(downTime, now, action, x, y, 0);
        event.setSource(InputDevice.SOURCE_TOUCHSCREEN);
        injectInputEvent(event);
        event.recycle();
    }

    private void injectKeyEvent(int action, int keyCode, long downTime) {
        injectKeyEvent(action, keyCode, downTime, 0);
    }

    private void injectKeyEvent(int action, int keyCode, long downTime, int metaState) {
        long now = SystemClock.uptimeMillis();
        KeyEvent event = new KeyEvent(downTime, now, action, keyCode, 0, metaState, 0, 0,
                KeyEvent.FLAG_FROM_SYSTEM, InputDevice.SOURCE_KEYBOARD);
        injectInputEvent(event);
    }

    private void injectInputEvent(android.view.InputEvent event) {
        if (mInputManager == null) {
            Log.e(TAG, "Input injection not available");
            return;
        }
        try {
            mInputManager.injectInputEvent(event,
                    InputManager.INJECT_INPUT_EVENT_MODE_ASYNC);
        } catch (SecurityException e) {
            Log.e(TAG, "Not allowed to inject input - check INJECT_EVENTS grant", e);
        } catch (Exception e) {
            Log.e(TAG, "Failed to inject input event", e);
        }
    }

    /** Tap at the given screen coordinates. */
    public void tap(float x, float y) {
        Log.i(TAG, "Tapping at (" + x + ", " + y + ")");
        long downTime = SystemClock.uptimeMillis();
        injectMotionEvent(MotionEvent.ACTION_DOWN, x, y, downTime);
        sleepCancellable(50);
        injectMotionEvent(MotionEvent.ACTION_UP, x, y, downTime);
        mLastTouchX = x;
        mLastTouchY = y;
        mLastTouchTime = downTime;
    }

    /**
     * Swipe gesture. When the abort flag is raised mid-gesture the swipe is
     * wound up early, but the closing ACTION_UP is always injected so the
     * system never keeps a touch stuck down.
     *
     * @return false when the gesture was aborted mid-way
     */
    public boolean swipe(float startX, float startY, float endX, float endY, long durationMs) {
        Log.i(TAG, "Swiping from (" + startX + ", " + startY + ") to (" + endX + ", " + endY + ")");
        long downTime = SystemClock.uptimeMillis();
        long steps = Math.max(1, durationMs / 50);
        float dx = (endX - startX) / steps;
        float dy = (endY - startY) / steps;

        injectMotionEvent(MotionEvent.ACTION_DOWN, startX, startY, downTime);
        boolean aborted = false;
        for (int i = 1; i <= steps; i++) {
            if (sleepCancellable(50)) {
                aborted = true;
                break;
            }
            injectMotionEvent(MotionEvent.ACTION_MOVE, startX + dx * i, startY + dy * i,
                    downTime);
        }
        injectMotionEvent(MotionEvent.ACTION_UP, aborted ? mLastTouchX : endX,
                aborted ? mLastTouchY : endY, downTime);
        float lastX = aborted ? startX + dx * Math.max(0, steps - 1) : endX;
        float lastY = aborted ? startY + dy * Math.max(0, steps - 1) : endY;
        mLastTouchX = lastX;
        mLastTouchY = lastY;
        mLastTouchTime = downTime;
        return !aborted;
    }

    /** Long press. */
    public void longPress(float x, float y, long durationMs) {
        Log.i(TAG, "Long press at (" + x + ", " + y + ") for " + durationMs + "ms");
        long downTime = SystemClock.uptimeMillis();
        injectMotionEvent(MotionEvent.ACTION_DOWN, x, y, downTime);
        sleepCancellable(durationMs);
        injectMotionEvent(MotionEvent.ACTION_UP, x, y, downTime);
        mLastTouchX = x;
        mLastTouchY = y;
        mLastTouchTime = downTime;
    }

    /** Press a key. */
    public void keyPress(int keyCode) {
        Log.i(TAG, "Pressing key code: " + keyCode);
        long downTime = SystemClock.uptimeMillis();
        injectKeyEvent(KeyEvent.ACTION_DOWN, keyCode, downTime);
        sleepCancellable(50);
        injectKeyEvent(KeyEvent.ACTION_UP, keyCode, downTime);
    }

    /**
     * Type text into the focused field.
     *
     * ASCII strings are typed through the virtual keyboard character map. Any
     * text containing non-ASCII characters (Chinese, emoji, ...) or very long
     * strings cannot be produced by that map - those are put on the clipboard
     * and pasted with Ctrl+V, which is what makes CJK search boxes usable.
     */
    public void textInput(String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        Log.i(TAG, "Inputting text of length " + text.length());
        if (!isAsciiText(text) || text.length() > 200) {
            if (pasteText(text)) {
                return;
            }
            Log.w(TAG, "Clipboard paste failed; falling back to key events");
        }
        try {
            KeyCharacterMap kcm = KeyCharacterMap.load(KeyCharacterMap.VIRTUAL_KEYBOARD);
            KeyEvent[] events = kcm.getEvents(text.toCharArray());
            if (events != null) {
                for (KeyEvent event : events) {
                    if (mAborted) {
                        return;
                    }
                    injectInputEvent(event);
                    if (sleepCancellable(20)) {
                        return;
                    }
                }
                return;
            }
        } catch (Exception e) {
            Log.w(TAG, "KeyCharacterMap typing failed, falling back to simple mapping", e);
        }
        // Fallback: basic A-Z, 0-9, space, enter
        for (char c : text.toCharArray()) {
            if (mAborted) {
                return;
            }
            int keyCode = getKeyCodeForChar(c);
            if (keyCode != 0) {
                keyPress(keyCode);
                if (sleepCancellable(30)) {
                    return;
                }
            }
        }
    }

    /** Scroll vertically at the given position. Negative deltaY scrolls up. */
    public void scroll(float x, float y, float deltaY) {
        Log.i(TAG, "Scrolling at (" + x + ", " + y + ") by " + deltaY);
        long downTime = SystemClock.uptimeMillis();
        float endY = y + deltaY;
        injectMotionEvent(MotionEvent.ACTION_DOWN, x, y, downTime);
        sleepCancellable(50);
        injectMotionEvent(MotionEvent.ACTION_MOVE, x, y + deltaY / 2, downTime);
        sleepCancellable(50);
        injectMotionEvent(MotionEvent.ACTION_MOVE, x, endY, downTime);
        sleepCancellable(50);
        injectMotionEvent(MotionEvent.ACTION_UP, x, endY, downTime);
        mLastTouchX = x;
        mLastTouchY = endY;
        mLastTouchTime = downTime;
    }

    private static boolean isAsciiText(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c < 0x20 || c > 0x7e) {
                return false;
            }
        }
        return true;
    }

    /**
     * Put the text on the clipboard and inject Ctrl+V into the focused field.
     * Text fields in Android handle the hardware Ctrl+V shortcut even when no
     * IME is attached; this bypasses the ASCII-only KeyCharacterMap limit.
     */
    private boolean pasteText(String text) {
        try {
            ClipboardManager clipboard = (ClipboardManager)
                    mParentService.getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) {
                Log.w(TAG, "No ClipboardManager");
                return false;
            }
            clipboard.setPrimaryClip(ClipData.newPlainText("AIPhoneAssistant", text));
            if (sleepCancellable(150)) {
                return true;
            }
            Log.i(TAG, "Pasting " + text.length() + " chars from the clipboard");
            long t = SystemClock.uptimeMillis();
            // The V events must carry META_CTRL_ON themselves: a held CTRL key
            // event does not automatically set the modifier state on the
            // following events, and without it the IME just types a "v".
            injectKeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_CTRL_LEFT, t);
            sleepCancellable(30);
            injectKeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_V, t,
                    KeyEvent.META_CTRL_ON);
            sleepCancellable(30);
            injectKeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_V, t,
                    KeyEvent.META_CTRL_ON);
            sleepCancellable(30);
            injectKeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_CTRL_LEFT, t);
            sleepCancellable(60);
            return true;
        } catch (Exception e) {
            Log.w(TAG, "Clipboard paste failed", e);
            return false;
        }
    }

    private int getKeyCodeForChar(char c) {
        if (c >= 'a' && c <= 'z') {
            return KeyEvent.KEYCODE_A + (c - 'a');
        }
        if (c >= 'A' && c <= 'Z') {
            // Same physical key; real typing needs shift, not handled in the fallback.
            return KeyEvent.KEYCODE_A + (c - 'A');
        }
        if (c >= '0' && c <= '9') {
            return KeyEvent.KEYCODE_0 + (c - '0');
        }
        if (c == ' ') {
            return KeyEvent.KEYCODE_SPACE;
        }
        if (c == '\n') {
            return KeyEvent.KEYCODE_ENTER;
        }
        return 0;
    }

    private static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Like {@link #sleep(long)} but wakes up early when the abort flag is set
     * or the thread is interrupted.
     *
     * @return true when the wait was cut short by the abort flag
     */
    private boolean sleepCancellable(long ms) {
        long deadline = SystemClock.uptimeMillis() + ms;
        while (!mAborted) {
            long remaining = deadline - SystemClock.uptimeMillis();
            if (remaining <= 0) {
                return false;
            }
            try {
                Thread.sleep(Math.min(remaining, 25));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return mAborted;
            }
        }
        return true;
    }

    public float getLastTouchX() {
        return mLastTouchX;
    }

    public float getLastTouchY() {
        return mLastTouchY;
    }

    public void release() {
        Log.i(TAG, "Releasing InputInjectionService");
        mInputManager = null;
    }

    /** Input action types. */
    public enum ActionType {
        TAP,
        SWIPE,
        LONG_PRESS,
        KEY_PRESS,
        TEXT_INPUT,
        SCROLL,
        DOUBLE_TAP,
        PINCH,
        WAIT,
        NONE
    }

    /** Represents an input action to be executed. */
    public static class InputAction {
        public ActionType type;
        public float x;
        public float y;
        public float endX;
        public float endY;
        public long durationMs;
        public int keyCode;
        public String text;
        public float deltaY;

        public static InputAction tap(float x, float y) {
            InputAction action = new InputAction();
            action.type = ActionType.TAP;
            action.x = x;
            action.y = y;
            return action;
        }

        public static InputAction swipe(float startX, float startY, float endX, float endY,
                long durationMs) {
            InputAction action = new InputAction();
            action.type = ActionType.SWIPE;
            action.x = startX;
            action.y = startY;
            action.endX = endX;
            action.endY = endY;
            action.durationMs = durationMs;
            return action;
        }

        public static InputAction longPress(float x, float y, long durationMs) {
            InputAction action = new InputAction();
            action.type = ActionType.LONG_PRESS;
            action.x = x;
            action.y = y;
            action.durationMs = durationMs;
            return action;
        }

        public static InputAction keyPress(int keyCode) {
            InputAction action = new InputAction();
            action.type = ActionType.KEY_PRESS;
            action.keyCode = keyCode;
            return action;
        }

        public static InputAction textInput(String text) {
            InputAction action = new InputAction();
            action.type = ActionType.TEXT_INPUT;
            action.text = text;
            return action;
        }

        public static InputAction scroll(float x, float y, float deltaY) {
            InputAction action = new InputAction();
            action.type = ActionType.SCROLL;
            action.x = x;
            action.y = y;
            action.deltaY = deltaY;
            return action;
        }

        public static InputAction waitAction(long durationMs) {
            InputAction action = new InputAction();
            action.type = ActionType.WAIT;
            action.durationMs = Math.min(Math.max(durationMs, 100), 5000);
            return action;
        }

        /**
         * Representative screen point for this action, used to position the overlay
         * marker at the location of injection. For tap/long-press this is the touch
         * point; for swipe/scroll it is the gesture start point. Returns null for
         * actions without a screen position.
         */
        public android.graphics.PointF getPrimaryPoint() {
            switch (type) {
                case TAP:
                case LONG_PRESS:
                case SWIPE:
                case SCROLL:
                    return new android.graphics.PointF(x, y);
                default:
                    return null;
            }
        }
    }
}
