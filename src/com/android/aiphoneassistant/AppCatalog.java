//
// Copyright (C) 2026 The AgentVisOS Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.android.aiphoneassistant;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
import android.content.pm.InstallSourceInfo;
import android.content.pm.LauncherApps;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.content.pm.ShortcutInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Process;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * On-device catalogue of the installed apps and how each one can be entered,
 * so the model can jump straight into an app instead of hunting for its icon.
 *
 * <h3>What the framework exposes, and how we get it</h3>
 *
 * <ul>
 *   <li><b>Identity / metadata</b> — {@link PackageManager#getInstalledPackages}
 *       with {@code GET_ACTIVITIES|GET_RECEIVERS|GET_SERVICES|GET_META_DATA}:
 *       label, package, version, uid, target/min SDK, system / enabled /
 *       debuggable flags, Play category, installer, process name.</li>
 *   <li><b>Launcher entry points</b> — {@code getLaunchIntentForPackage} /
 *       {@code getLeanbackLaunchIntentForPackage}, plus MAIN/LAUNCHER and
 *       MAIN/LEANBACK_LAUNCHER components found on the activity list.</li>
 *   <li><b>URI / deep-link handlers</b> — the platform deliberately does
 *       <em>not</em> expose a component's raw {@code IntentFilter} array
 *       ({@code ComponentInfo} has no such field, public or hidden), so we
 *       probe: for every scheme we care about we ask
 *       {@code queryIntentActivities(VIEW scheme://)} and read back the matched
 *       {@link ResolveInfo#filter} — which does carry the scheme, host and
 *       path prefix that matched. That yields the same "which app opens which
 *       URI" map the OS uses to route links.</li>
 *   <li><b>Share targets</b> — probes for {@code ACTION_SEND} with
 *       {@code text/plain} and {@code image/*}, and {@code ACTION_SENDTO} for
 *       {@code smsto:} / {@code mailto:}.</li>
 *   <li><b>Shortcuts</b> — {@link LauncherApps#getShortcuts} when this process
 *       is allowed to host shortcuts (a privileged system app usually is).</li>
 * </ul>
 *
 * <p>Two renderings exist on purpose: {@link App#compactLine()} is what the
 * model sees (dense, one line per app) and {@link #toJson()} is the full dump
 * for a human or for tooling. The model never gets the full dump: a few
 * hundred apps of raw component names would burn the context for no benefit.
 */
public final class AppCatalog {

    private static final String TAG = "AIPhoneAssistant.Apps";

    private static final int MAX_SHORTCUTS_PER_APP = 8;
    private static final int MAX_LAUNCH_COMPONENTS = 4;
    private static final int MAX_URI_HANDLERS_PER_APP = 24;
    private static final int MAX_SHARE_MIMES_PER_APP = 4;

    /**
     * Schemes we probe. The first block is what the platform itself routes
     * (web, telephony, mail, maps, store); the second is the set of Chinese
     * app schemes worth knowing about — a superset would cost a few hundred ms
     * more for little extra coverage.
     */
    private static final String[] PROBE_SCHEMES = {
            // platform / ubiquitous
            "http", "https", "tel", "sms", "smsto", "mailto", "geo", "market",
            // messaging / social
            "weixin", "wechat", "qq", "mqq", "sinaweibo", "weibo", "zhihu",
            "douyin", "snssdk1128", "xiaohongshu",
            // commerce / services
            "taobao", "tmall", "openapp.jdmobile", "alipays", "alipay",
            "dianping", "meituan", "didi", "ctrip", "bilibili", "xhs",
    };

    private static final Object LOCK = new Object();
    private static volatile AppCatalog sInstance;
    private static volatile Thread sWarmUp;

    // ------------------------------------------------------------------
    // Model
    // ------------------------------------------------------------------

    /** One "this app can open scheme://…[@path]" fact. */
    public static final class UriHandler {
        public final String scheme;
        public final String host;
        public final String path;
        public final String component;

        UriHandler(String scheme, String host, String path, String component) {
            this.scheme = scheme;
            this.host = host;
            this.path = path;
            this.component = component;
        }

        /** A URI that is very likely to be accepted by that app. */
        public String sampleUri() {
            if (scheme == null || scheme.isEmpty()) {
                return null;
            }
            StringBuilder sb = new StringBuilder(scheme).append("://");
            if (host != null && !host.isEmpty() && !"*".equals(host)) {
                sb.append(host);
            }
            String p = path;
            if (p != null && !p.isEmpty() && !"*".equals(p)) {
                if (p.contains("*")) {
                    p = p.substring(0, p.indexOf('*'));
                }
                if (p.contains(".*")) {
                    p = p.substring(0, p.indexOf(".*"));
                }
                if (!p.startsWith("/") && sb.charAt(sb.length() - 1) != '/') {
                    sb.append('/');
                }
                sb.append(p);
            }
            String out = sb.toString();
            return out.length() > 3 ? out : null;
        }

        String compact() {
            StringBuilder sb = new StringBuilder();
            if (scheme != null && !scheme.isEmpty()) {
                sb.append(scheme);
                if (host != null && !host.isEmpty() && !"*".equals(host)) {
                    sb.append("://").append(host);
                }
                if (path != null && !path.isEmpty() && !"/*".equals(path)) {
                    sb.append(path);
                }
            }
            return sb.toString();
        }
    }

    /** An {@code ACTION_SEND}/{@code ACTION_SENDTO} target. */
    public static final class ShareTarget {
        public final String mime;
        public final String component;

        ShareTarget(String mime, String component) {
            this.mime = mime;
            this.component = component;
        }
    }

    /** A launcher shortcut (long-press entry point), when readable. */
    public static final class Shortcut {
        public final String id;
        public final String shortLabel;
        public final String longLabel;

        Shortcut(String id, String shortLabel, String longLabel) {
            this.id = id;
            this.shortLabel = shortLabel;
            this.longLabel = longLabel;
        }
    }

    /** Everything known about one installed app. */
    public static final class App {
        public String pkg = "";
        public String label = "";
        public String versionName = "";
        public long versionCode;
        public int targetSdk;
        public int minSdk;
        public int uid;
        public boolean system;
        public boolean enabled = true;
        public boolean debuggable;
        public boolean hasLauncher;
        public String category = "";
        public String installer = "";
        public String process = "";

        public final List<String> launchComponents = new ArrayList<>();
        public final List<UriHandler> uriHandlers = new ArrayList<>();
        public final List<ShareTarget> shareTargets = new ArrayList<>();
        public final List<Shortcut> shortcuts = new ArrayList<>();

        private String haystack = "";

        void seal() {
            StringBuilder sb = new StringBuilder();
            sb.append(label).append(' ').append(pkg);
            for (String c : launchComponents) {
                sb.append(' ').append(c);
            }
            for (UriHandler h : uriHandlers) {
                sb.append(' ').append(h.scheme);
                if (h.host != null) {
                    sb.append(' ').append(h.host);
                }
            }
            haystack = sb.toString().toLowerCase(Locale.ROOT);
        }

        public boolean canOpenUris() {
            return !uriHandlers.isEmpty();
        }

        /** Compact, model-facing one-liner. */
        public String compactLine() {
            StringBuilder sb = new StringBuilder();
            sb.append(label.isEmpty() ? pkg : label);
            sb.append(" | ").append(pkg);
            if (!versionName.isEmpty()) {
                sb.append(" | v").append(versionName);
            }
            if (hasLauncher) {
                sb.append(" | launch");
            }
            if (!uriHandlers.isEmpty()) {
                sb.append(" | uris:");
                Set<String> seen = new LinkedHashSet<>();
                int n = 0;
                for (UriHandler h : uriHandlers) {
                    String c = h.compact();
                    if (c.isEmpty() || !seen.add(c)) {
                        continue;
                    }
                    if (n > 0) {
                        sb.append(',');
                    }
                    sb.append(c);
                    if (++n >= 5) {
                        sb.append(",…");
                        break;
                    }
                }
            }
            if (!shareTargets.isEmpty()) {
                Set<String> mimes = new LinkedHashSet<>();
                for (ShareTarget t : shareTargets) {
                    if (t.mime != null && !t.mime.isEmpty() && !"*/*".equals(t.mime)) {
                        mimes.add(t.mime);
                    }
                    if (mimes.size() >= MAX_SHARE_MIMES_PER_APP) {
                        break;
                    }
                }
                if (!mimes.isEmpty()) {
                    sb.append(" | share:").append(String.join(",", mimes));
                }
            }
            if (!shortcuts.isEmpty()) {
                sb.append(" | sc:");
                int n = 0;
                for (Shortcut s : shortcuts) {
                    String l = s.shortLabel != null && !s.shortLabel.isEmpty()
                            ? s.shortLabel : s.longLabel;
                    if (l == null || l.isEmpty()) {
                        continue;
                    }
                    if (n > 0) {
                        sb.append(',');
                    }
                    sb.append(l);
                    if (++n >= 3) {
                        break;
                    }
                }
            }
            String out = sb.toString();
            return out.length() <= 240 ? out : out.substring(0, 237) + "…";
        }

        /**
         * Resolve one of this app's launcher shortcuts by id or visible label
         * (exact match first, then case-insensitive contains). Returns null
         * when the app exposes no matching shortcut.
         */
        public String resolveShortcutId(String query) {
            if (query == null || query.trim().isEmpty()) {
                return null;
            }
            String q = query.trim().toLowerCase(java.util.Locale.ROOT);
            for (Shortcut s : shortcuts) {
                if (s.id != null && s.id.toLowerCase(java.util.Locale.ROOT).equals(q)) {
                    return s.id;
                }
            }
            for (Shortcut s : shortcuts) {
                if (s.shortLabel != null
                        && s.shortLabel.toLowerCase(java.util.Locale.ROOT).equals(q)) {
                    return s.id;
                }
                if (s.longLabel != null
                        && s.longLabel.toLowerCase(java.util.Locale.ROOT).equals(q)) {
                    return s.id;
                }
            }
            for (Shortcut s : shortcuts) {
                if ((s.id != null && s.id.toLowerCase(java.util.Locale.ROOT).contains(q))
                        || (s.shortLabel != null
                                && s.shortLabel.toLowerCase(java.util.Locale.ROOT).contains(q))
                        || (s.longLabel != null
                                && s.longLabel.toLowerCase(java.util.Locale.ROOT).contains(q))) {
                    return s.id;
                }
            }
            return null;
        }

        /** Rich, human-facing dump for the debug view. */
        public JSONObject toJson() throws org.json.JSONException {
            JSONObject o = new JSONObject();
            o.put("package", pkg);
            o.put("label", label);
            o.put("versionName", versionName);
            o.put("versionCode", versionCode);
            o.put("targetSdk", targetSdk);
            o.put("minSdk", minSdk);
            o.put("uid", uid);
            o.put("system", system);
            o.put("enabled", enabled);
            o.put("debuggable", debuggable);
            o.put("category", category);
            o.put("installer", installer);
            o.put("process", process);
            o.put("hasLauncher", hasLauncher);
            if (!launchComponents.isEmpty()) {
                o.put("launchComponents", new JSONArray(launchComponents));
            }
            JSONArray links = new JSONArray();
            for (UriHandler h : uriHandlers) {
                JSONObject l = new JSONObject();
                l.put("scheme", h.scheme);
                if (h.host != null) l.put("host", h.host);
                if (h.path != null) l.put("path", h.path);
                if (h.component != null) l.put("component", h.component);
                String sample = h.sampleUri();
                if (sample != null) l.put("sample", sample);
                links.put(l);
            }
            if (links.length() > 0) {
                o.put("uriHandlers", links);
            }
            JSONArray shares = new JSONArray();
            for (ShareTarget t : shareTargets) {
                JSONObject s = new JSONObject();
                s.put("mime", t.mime);
                s.put("component", t.component);
                shares.put(s);
            }
            if (shares.length() > 0) {
                o.put("shareTargets", shares);
            }
            JSONArray scs = new JSONArray();
            for (Shortcut s : shortcuts) {
                JSONObject x = new JSONObject();
                x.put("id", s.id);
                x.put("shortLabel", s.shortLabel);
                x.put("longLabel", s.longLabel);
                scs.put(x);
            }
            if (scs.length() > 0) {
                o.put("shortcuts", scs);
            }
            return o;
        }
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    private final List<App> mApps;
    private final Map<String, App> mByPackage;
    private final long mBuiltAtMillis;
    private final int mPackageCount;

    private AppCatalog(List<App> apps, Map<String, App> byPackage, int packageCount) {
        mApps = apps;
        mByPackage = byPackage;
        mPackageCount = packageCount;
        mBuiltAtMillis = System.currentTimeMillis();
    }

    public static AppCatalog peek() {
        return sInstance;
    }

    /** Build now (blocking). Serialised; the result replaces the cache. */
    public static AppCatalog get(Context context) {
        AppCatalog cached = sInstance;
        if (cached != null) {
            return cached;
        }
        synchronized (LOCK) {
            if (sInstance != null) {
                return sInstance;
            }
            long t0 = System.currentTimeMillis();
            AppCatalog built = build(context.getApplicationContext());
            Log.i(TAG, "App catalog built: " + built.mApps.size() + " apps from "
                    + built.mPackageCount + " packages in "
                    + (System.currentTimeMillis() - t0) + " ms");
            sInstance = built;
            return built;
        }
    }

    /** Build off the caller's thread; the capture loop must never pay this. */
    public static void warmUp(final Context context) {
        if (sInstance != null) {
            return;
        }
        synchronized (LOCK) {
            if (sWarmUp != null && sWarmUp.isAlive()) {
                return;
            }
            sWarmUp = new Thread(() -> {
                try {
                    get(context);
                } catch (Exception e) {
                    Log.w(TAG, "App catalog warm-up failed", e);
                }
            }, "aiphone-app-catalog");
            sWarmUp.setDaemon(true);
            sWarmUp.start();
        }
    }

    public static void invalidate() {
        synchronized (LOCK) {
            sInstance = null;
        }
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    private static AppCatalog build(Context context) {
        PackageManager pm = context.getPackageManager();
        // Only activities are needed: the per-component IntentFilters the
        // platform refuses to expose were replaced by scheme probes, and
        // services/receivers/meta-data would just make the parcel bigger.
        int flags = PackageManager.GET_ACTIVITIES;

        List<PackageInfo> packages;
        try {
            packages = pm.getInstalledPackages(flags);
        } catch (Exception e) {
            Log.w(TAG, "getInstalledPackages(flags) failed, retrying with 0", e);
            packages = pm.getInstalledPackages(0);
        }

        List<App> apps = new ArrayList<>(packages.size());
        Map<String, App> byPackage = new LinkedHashMap<>();
        for (PackageInfo pi : packages) {
            if (pi == null || pi.applicationInfo == null || pi.packageName == null) {
                continue;
            }
            // Runtime resource overlays (RRO) are not apps: they have no
            // launcher entry and their label usually falls back to the package
            // name (`com.android.settings.auto_generated_rro_product__`), which
            // pollutes list_apps results and can shadow the real app when the
            // model asks to open a package name. Skip them entirely.
            if (pi.applicationInfo.isResourceOverlay()) {
                continue;
            }
            try {
                App app = fromPackage(pm, pi);
                if (app != null) {
                    apps.add(app);
                    byPackage.put(app.pkg, app);
                }
            } catch (Exception e) {
                Log.w(TAG, "Skipping package " + pi.packageName, e);
            }
        }

        // Probes are what make the catalogue useful: they turn the platform's
        // link routing into per-app facts.
        probeUriHandlers(pm, byPackage);
        probeShareTargets(pm, byPackage);
        collectShortcuts(context, apps);

        for (App app : apps) {
            app.seal();
        }
        Collections.sort(apps, Comparator
                .comparing((App a) -> a.label == null ? "" : a.label.toLowerCase(Locale.ROOT))
                .thenComparing(a -> a.pkg));
        return new AppCatalog(apps, byPackage, packages.size());
    }

    private static App fromPackage(PackageManager pm, PackageInfo pi) {
        ApplicationInfo ai = pi.applicationInfo;
        if (ai == null) {
            return null;
        }
        App app = new App();
        app.pkg = pi.packageName == null ? "" : pi.packageName;
        if (app.pkg.isEmpty()) {
            return null;
        }
        CharSequence label = null;
        try {
            label = ai.loadLabel(pm);
        } catch (Exception ignored) {
            // fall back to the package name
        }
        app.label = label == null || label.length() == 0 ? app.pkg : label.toString();
        app.versionName = pi.versionName == null ? "" : pi.versionName;
        app.versionCode = Build.VERSION.SDK_INT >= 28 ? pi.getLongVersionCode() : pi.versionCode;
        app.targetSdk = ai.targetSdkVersion;
        app.minSdk = ai.minSdkVersion;
        app.uid = ai.uid;
        app.enabled = ai.enabled;
        app.system = (ai.flags & ApplicationInfo.FLAG_SYSTEM) != 0;
        app.debuggable = (ai.flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        app.process = ai.processName == null ? app.pkg : ai.processName;
        if (Build.VERSION.SDK_INT >= 26) {
            app.category = categoryName(ai.category);
        }
        app.installer = installerOf(pm, app.pkg);

        Intent launch = pm.getLaunchIntentForPackage(app.pkg);
        if (launch != null && launch.getComponent() != null) {
            app.hasLauncher = true;
            app.launchComponents.add(launch.getComponent().flattenToShortString());
        }
        if (!app.hasLauncher) {
            Intent leanback = pm.getLeanbackLaunchIntentForPackage(app.pkg);
            if (leanback != null && leanback.getComponent() != null) {
                app.hasLauncher = true;
                app.launchComponents.add(leanback.getComponent().flattenToShortString());
            }
        }
        return app;
    }

    /**
     * Probe every scheme in {@link #PROBE_SCHEMES} and attribute the handlers
     * to their apps. {@link ResolveInfo#filter} is the matched filter, so the
     * host/path it carries is exactly the part the app declared.
     */
    private static void probeUriHandlers(PackageManager pm, Map<String, App> byPackage) {
        for (String scheme : PROBE_SCHEMES) {
            Intent probe;
            try {
                probe = new Intent(Intent.ACTION_VIEW, Uri.parse(scheme + "://"));
            } catch (Exception e) {
                continue;
            }
            List<ResolveInfo> matches;
            try {
                matches = pm.queryIntentActivities(probe,
                        PackageManager.MATCH_ALL | PackageManager.MATCH_DEFAULT_ONLY);
                if (matches == null || matches.isEmpty()) {
                    matches = pm.queryIntentActivities(probe, PackageManager.MATCH_ALL);
                }
            } catch (Exception e) {
                Log.d(TAG, "Probe failed for scheme " + scheme + ": " + e.getMessage());
                continue;
            }
            if (matches == null) {
                continue;
            }
            for (ResolveInfo ri : matches) {
                if (ri.activityInfo == null) {
                    continue;
                }
                App app = byPackage.get(ri.activityInfo.packageName);
                if (app == null || app.uriHandlers.size() >= MAX_URI_HANDLERS_PER_APP) {
                    continue;
                }
                String host = null;
                String path = null;
                android.content.IntentFilter filter = ri.filter;
                if (filter != null) {
                    try {
                        if (filter.countDataAuthorities() > 0) {
                            android.content.IntentFilter.AuthorityEntry entry =
                                    filter.getDataAuthority(0);
                            if (entry != null && entry.getHost() != null
                                    && !"*".equals(entry.getHost())) {
                                host = entry.getHost();
                            }
                        }
                        if (filter.countDataPaths() > 0) {
                            android.os.PatternMatcher matcher = filter.getDataPath(0);
                            if (matcher != null) {
                                path = matcher.getPath();
                            }
                        }
                        if (path == null && filter.countDataSchemeSpecificParts() > 0) {
                            android.os.PatternMatcher matcher =
                                    filter.getDataSchemeSpecificPart(0);
                            if (matcher != null) {
                                path = matcher.getPath();
                            }
                        }
                    } catch (Exception ignored) {
                        // Filter introspection is best-effort.
                    }
                }
                String component = ri.activityInfo.name;
                boolean duplicate = false;
                for (UriHandler existing : app.uriHandlers) {
                    if (existing.scheme.equals(scheme)
                            && java.util.Objects.equals(existing.host, host)
                            && java.util.Objects.equals(existing.path, path)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    app.uriHandlers.add(new UriHandler(scheme, host, path, component));
                }
            }
        }
    }

    /** Probe the share sheet: who accepts text, who accepts images, who sends SMS/mail. */
    private static void probeShareTargets(PackageManager pm, Map<String, App> byPackage) {
        probeShare(pm, byPackage, new Intent(Intent.ACTION_SEND).setType("text/plain"), "text/plain");
        probeShare(pm, byPackage, new Intent(Intent.ACTION_SEND).setType("image/*"), "image/*");
        probeShare(pm, byPackage, new Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:")), "smsto:");
        probeShare(pm, byPackage, new Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:")), "mailto:");
    }

    private static void probeShare(PackageManager pm, Map<String, App> byPackage,
            Intent intent, String label) {
        List<ResolveInfo> matches;
        try {
            matches = pm.queryIntentActivities(intent, PackageManager.MATCH_ALL);
        } catch (Exception e) {
            return;
        }
        if (matches == null) {
            return;
        }
        for (ResolveInfo ri : matches) {
            if (ri.activityInfo == null) {
                continue;
            }
            App app = byPackage.get(ri.activityInfo.packageName);
            if (app == null || app.shareTargets.size() >= MAX_SHARE_MIMES_PER_APP) {
                continue;
            }
            boolean duplicate = false;
            for (ShareTarget existing : app.shareTargets) {
                if (existing.mime.equals(label)) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                app.shareTargets.add(new ShareTarget(label, ri.activityInfo.name));
            }
        }
    }

    /** Manifest shortcuts of every app, when this process may read them. */
    private static void collectShortcuts(Context context, List<App> apps) {
        if (Build.VERSION.SDK_INT < 25) {
            return;
        }
        LauncherApps la;
        try {
            la = (LauncherApps) context.getSystemService(Context.LAUNCHER_APPS_SERVICE);
            if (la == null || !la.hasShortcutHostPermission()) {
                Log.i(TAG, "Shortcut host permission not held; skipping shortcuts");
                return;
            }
        } catch (Exception e) {
            return;
        }
        android.os.UserHandle user = Process.myUserHandle();
        for (App app : apps) {
            if (!app.hasLauncher) {
                continue;
            }
            try {
                LauncherApps.ShortcutQuery query = new LauncherApps.ShortcutQuery()
                        .setPackage(app.pkg)
                        .setQueryFlags(LauncherApps.ShortcutQuery.FLAG_MATCH_MANIFEST
                                | LauncherApps.ShortcutQuery.FLAG_MATCH_DYNAMIC);
                List<ShortcutInfo> list = la.getShortcuts(query, user);
                if (list == null) {
                    continue;
                }
                for (ShortcutInfo si : list) {
                    if (si == null || app.shortcuts.size() >= MAX_SHORTCUTS_PER_APP) {
                        break;
                    }
                    CharSequence shortLabel = si.getShortLabel();
                    CharSequence longLabel = si.getLongLabel();
                    app.shortcuts.add(new Shortcut(
                            si.getId(),
                            shortLabel == null ? "" : shortLabel.toString(),
                            longLabel == null ? "" : longLabel.toString()));
                }
            } catch (Exception e) {
                Log.d(TAG, "Shortcuts unavailable for " + app.pkg + ": " + e.getMessage());
            }
        }
    }

    private static String installerOf(PackageManager pm, String pkg) {
        try {
            if (Build.VERSION.SDK_INT >= 30) {
                InstallSourceInfo info = pm.getInstallSourceInfo(pkg);
                if (info != null && info.getInstallingPackageName() != null) {
                    return info.getInstallingPackageName();
                }
            } else {
                String installer = pm.getInstallerPackageName(pkg);
                if (installer != null) {
                    return installer;
                }
            }
        } catch (Exception ignored) {
            // pass
        }
        return "";
    }

    private static String categoryName(int category) {
        switch (category) {
            case ApplicationInfo.CATEGORY_GAME: return "game";
            case ApplicationInfo.CATEGORY_AUDIO: return "audio";
            case ApplicationInfo.CATEGORY_VIDEO: return "video";
            case ApplicationInfo.CATEGORY_IMAGE: return "image";
            case ApplicationInfo.CATEGORY_SOCIAL: return "social";
            case ApplicationInfo.CATEGORY_NEWS: return "news";
            case ApplicationInfo.CATEGORY_MAPS: return "maps";
            case ApplicationInfo.CATEGORY_PRODUCTIVITY: return "productivity";
            case ApplicationInfo.CATEGORY_ACCESSIBILITY: return "accessibility";
            default: return "";
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    public int size() {
        return mApps.size();
    }

    public List<App> apps() {
        return Collections.unmodifiableList(mApps);
    }

    public App byPackage(String pkg) {
        return pkg == null ? null : mByPackage.get(pkg);
    }

    public long builtAtMillis() {
        return mBuiltAtMillis;
    }

    public List<String> compactLines() {
        List<String> out = new ArrayList<>(mApps.size());
        for (App a : mApps) {
            out.add(a.compactLine());
        }
        return out;
    }

    /**
     * Rank apps against a free-form query (display name, package, scheme).
     * An empty query returns the launchable apps, alphabetically.
     */
    public List<App> search(String query, int limit) {
        String q = query == null ? "" : query.trim().toLowerCase(Locale.ROOT);
        List<App> result = new ArrayList<>();
        if (q.isEmpty()) {
            for (App a : mApps) {
                if (a.hasLauncher && result.size() < limit) {
                    result.add(a);
                }
            }
            return result;
        }
        // 0 exact label, 1 label prefix, 2 label contains,
        // 3 package exact/prefix, 4 anything else in the haystack
        List<int[]> scores = new ArrayList<>();
        for (int i = 0; i < mApps.size(); i++) {
            App a = mApps.get(i);
            String label = a.label.toLowerCase(Locale.ROOT);
            int score;
            if (label.equals(q)) {
                score = 0;
            } else if (label.startsWith(q)) {
                score = 1;
            } else if (label.contains(q)) {
                score = 2;
            } else if (a.pkg.equals(q) || a.pkg.toLowerCase(Locale.ROOT).startsWith(q)) {
                score = 3;
            } else if (a.haystack.contains(q)) {
                score = 4;
            } else {
                continue;
            }
            scores.add(new int[]{score, i});
        }
        scores.sort((x, y) -> {
            if (x[0] != y[0]) {
                return x[0] - y[0];
            }
            // On equal text scores prefer an app that can actually be launched:
            // helper/service packages with a label that happens to match must
            // not shadow the real app.
            App ax = mApps.get(x[1]);
            App ay = mApps.get(y[1]);
            if (ax.hasLauncher != ay.hasLauncher) {
                return ax.hasLauncher ? -1 : 1;
            }
            return x[1] - y[1];
        });
        for (int[] s : scores) {
            if (result.size() >= limit) {
                break;
            }
            result.add(mApps.get(s[1]));
        }
        return result;
    }

    /** Resolve one app by exact label, then by unique prefix/contains match. */
    public App resolve(String query) {
        if (query == null || query.trim().isEmpty()) {
            return null;
        }
        // Package names are unambiguous identifiers: answer them exactly before
        // falling back to fuzzy label matching (otherwise an overlay/helper
        // package whose label happens to contain the query can win).
        App exactPkg = byPackage(query.trim());
        if (exactPkg != null) {
            return exactPkg;
        }
        List<App> matches = search(query, 8);
        if (matches.isEmpty()) {
            return null;
        }
        App best = matches.get(0);
        String q = query.trim().toLowerCase(Locale.ROOT);
        String label = best.label.toLowerCase(Locale.ROOT);
        if (!label.equals(q) && !label.startsWith(q)
                && !best.pkg.equalsIgnoreCase(query)
                && matches.size() > 1) {
            // "聊天" hits several labels; refuse instead of launching at random.
            App second = matches.get(1);
            if (second.label.toLowerCase(Locale.ROOT).contains(q)) {
                return null;
            }
        }
        return best;
    }

    /** Compact catalogue text for the model, capped by {@code limit} lines. */
    public String describeForPrompt(String query, int limit) {
        List<App> matches = search(query, limit);
        if (matches.isEmpty()) {
            return "No installed app matches \"" + query + "\".";
        }
        StringBuilder sb = new StringBuilder();
        sb.append(matches.size()).append(" app(s)");
        if (query != null && !query.trim().isEmpty()) {
            sb.append(" matching \"").append(query.trim()).append('"');
        }
        sb.append(" out of ").append(mApps.size()).append(" installed:\n");
        for (App a : matches) {
            sb.append("- ").append(a.compactLine()).append('\n');
        }
        return sb.toString();
    }

    /** Full JSON dump (debug screen / tooling). */
    public String toJson() {
        try {
            JSONArray arr = new JSONArray();
            for (App a : mApps) {
                arr.put(a.toJson());
            }
            JSONObject root = new JSONObject();
            root.put("builtAt", mBuiltAtMillis);
            root.put("count", mApps.size());
            root.put("apps", arr);
            return root.toString();
        } catch (org.json.JSONException e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }

    // ------------------------------------------------------------------
    // Launching
    // ------------------------------------------------------------------

    /**
     * Build a launch intent for {@code app}, optionally for a specific URI.
     *
     * @return the intent, or null when neither the URI nor the launcher entry
     *         resolves (the caller reports the failure to the model).
     */
    public static Intent buildLaunchIntent(Context context, App app, String uri) {
        PackageManager pm = context.getPackageManager();
        if (uri != null && !uri.trim().isEmpty()) {
            String trimmed = uri.trim();
            try {
                Intent view = new Intent(Intent.ACTION_VIEW, Uri.parse(trimmed));
                view.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                // Prefer a handler inside the requested app; fall back to any
                // handler (e.g. the browser for an https link).
                view.setPackage(app.pkg);
                List<ResolveInfo> inApp = pm.queryIntentActivities(view, 0);
                if (inApp != null && !inApp.isEmpty()) {
                    return view;
                }
                Intent any = new Intent(Intent.ACTION_VIEW, Uri.parse(trimmed));
                any.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                if (!pm.queryIntentActivities(any, 0).isEmpty()) {
                    return any;
                }
            } catch (Exception e) {
                Log.w(TAG, "Bad uri " + uri + ": " + e.getMessage());
            }
        }

        Intent launch = pm.getLaunchIntentForPackage(app.pkg);
        if (launch == null) {
            launch = pm.getLeanbackLaunchIntentForPackage(app.pkg);
        }
        if (launch == null) {
            for (String component : app.launchComponents) {
                ComponentName cn = ComponentName.unflattenFromString(component);
                if (cn != null) {
                    launch = new Intent(Intent.ACTION_MAIN)
                            .addCategory(Intent.CATEGORY_LAUNCHER)
                            .setComponent(cn);
                    break;
                }
            }
        }
        if (launch != null) {
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        }
        return launch;
    }

    /** The compact line for one package, or null. */
    public String compactFor(String pkg) {
        App a = byPackage(pkg);
        return a == null ? null : a.compactLine();
    }
}
