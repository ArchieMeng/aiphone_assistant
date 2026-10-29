#!/bin/bash
# Fetch the optional Qualcomm NPU runtime for the on-device backend:
#
#   1) libLiteRtDispatch_Qualcomm.so - prebuilt LiteRT NPU dispatch library from
#      the LiteRT-LM 0.18.0 Android native bundle (LiteRT is Apache-2.0;
#      checksum pinned below). The upstream AAR does not ship it yet, see
#      https://github.com/google-ai-edge/LiteRT/issues/6889 .
#   2) QNN/QAIRT runtime libraries + the signed Hexagon HTP skel, pulled from
#      the TARGET DEVICE itself (/odm/lib64/aiframe on OnePlus/SM8650). These
#      are Qualcomm proprietary files and are NOT redistributed by this repo:
#      they are copied from the owner's own device for their own ROM build.
#
# The libraries must come from the same QAIRT version as the device's QNN
# runtime; this device ships QAIRT 2.34.0 under /odm/lib64/aiframe (the default
# /odm/lib64/libQnnHtp.so is 2.13.1 and is too old).
#
# Usage: tools/fetch-npu-libs.sh [--adb "adb -s SERIAL"]
# Then:  $(call soong_config_set,aiphoneassistant,with_npu,true)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.cache/litertlm"
LIBS="$ROOT/libs/litertlm/arm64-v8a"
ADB="${ADB:-adb}"
DISPATCH_URL="https://github.com/DenisovAV/flutter_edge_ai/releases/download/native-v0.18.0-h/litertlm-android_arm64.tar.gz"
DISPATCH_TAR_SHA="d7ed14f761905cfe50bc06e3693e9abf934a96584c869410a78dfaef26080e62"
DISPATCH_SO_SHA="59c95fdc73685a9d9b209d3df2df60b09c1f2b0c513dcb789c10eb8444038105"

[ "$#" -gt 0 ] && [ "$1" = "--adb" ] && ADB="${2:?}"
mkdir -p "$CACHE" "$LIBS"

echo "=== [1/2] LiteRT NPU dispatch library ==="
TAR="$CACHE/litertlm-android_arm64.tar.gz"
if [ ! -f "$TAR" ] || ! echo "$DISPATCH_TAR_SHA  $TAR" | sha256sum -c --quiet 2>/dev/null; then
    echo "Downloading dispatch bundle"
    curl -fsSL -o "$TAR.tmp" "$DISPATCH_URL"
    echo "$DISPATCH_TAR_SHA  $TAR.tmp" | sha256sum -c --quiet || {
        echo "sha256 mismatch: dispatch bundle" >&2; exit 1; }
    mv "$TAR.tmp" "$TAR"
fi
tar -xzf "$TAR" -C "$CACHE" ./libLiteRtDispatch_Qualcomm.so
echo "$DISPATCH_SO_SHA  $CACHE/libLiteRtDispatch_Qualcomm.so" | sha256sum -c --quiet
cp -f "$CACHE/libLiteRtDispatch_Qualcomm.so" "$LIBS/"

echo "=== [2/2] QNN/QAIRT runtime from the device ($($ADB shell getprop ro.product.device 2>/dev/null | tr -d '\\r')) ==="
# Prefer the newest QAIRT bundle on the device; fail loudly if absent.
QNN_SRC="${QNN_SRC:-/odm/lib64/aiframe}"
SKEL_SRC="${SKEL_SRC:-$QNN_SRC/cdsp/signed}"
for f in libQnnHtp.so libQnnSystem.so libQnnHtpV75Stub.so libQnnHtpPrepare.so; do
    $ADB pull "$QNN_SRC/$f" "$LIBS/$f" >/dev/null
done
$ADB pull "$SKEL_SRC/libQnnHtpV75Skel.so" "$LIBS/libQnnHtpV75Skel.so" >/dev/null

sha256sum "$LIBS"/*.so > "$LIBS/NPU-LIBS-CHECKSUMS.txt"
echo "--- QNN files copied (device QNN version: $($ADB shell strings "$QNN_SRC/libQnnHtp.so" 2>/dev/null | grep -m1 -E '^v[0-9]+\\.' || echo unknown)) ---"
cat "$LIBS/NPU-LIBS-CHECKSUMS.txt"

# Regenerate libs/litertlm/Android.bp with the NPU prebuilt modules.
echo "=== regenerating libs/litertlm/Android.bp ==="
bash "$ROOT/tools/fetch-litertlm.sh"
echo
echo "NPU libraries ready. Enable with:"
echo "  \$(call soong_config_set,aiphoneassistant,with_npu,true)"
echo "NOTE: QAIRT dispatch/QNN versions must match the device QNN runtime;"
echo "      model must be an NPU-compiled (or NPU-JIT) .litertlm, e.g."
echo "      gemma-4-E2B-it_NPU_Uncompiled.litertlm."
