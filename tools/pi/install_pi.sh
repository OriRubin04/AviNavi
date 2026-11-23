#!/usr/bin/env bash
# AviNavi installer for a Raspberry Pi 5 (64-bit Raspberry Pi OS).
#
# Get the project onto the Pi first, either by copying the folder (USB stick)
# or with git (it asks for your GitHub login if the repository is private):
#     git clone -b claude/gps-denied-visual-localization-5oqtfv https://github.com/OriRubin04/AviNavi.git
#     cd AviNavi
#
# Then run from the project folder on the Pi:
#     bash tools/pi/install_pi.sh              # install + checks + report
#     bash tools/pi/install_pi.sh --dry-run    # only print what it would do
#
# Options:
#     --dry-run      print every step, change nothing (also works off a Pi)
#     --skip-tests   do not run the test suite at the end (saves a few minutes)
#     --with-maps    also install rasterio/pyproj (needed only to build map
#                    tiles from a GeoTIFF on the Pi, e.g. the UAV-VisLoc benchmark)
#     --no-apt       skip the one sudo step (system packages already installed)
#     --no-pin       install the newest torch/torchvision instead of the pinned ones
#
# What it changes: system packages via apt (the only sudo step), a .venv folder
# inside the project, and the model weights in ~/.cache/torch.
# What it never touches: config.txt, raspi-config, swap, CPU governor, user
# groups, serial-port settings. Those are reported, not changed.
#
# At the end it writes pi_install_report.txt in the project folder. Send that
# file back.
set -euo pipefail

# ---- versions measured on the laptop (keep the Pi on the same code) ---------
TORCH_VERSION="2.15.0"
TORCHVISION_VERSION="0.29.2"
# The SuperPoint resize=None fix (commit 7bd3209) relies on this LightGlue code.
LIGHTGLUE_REF="5a650c444718caffc2669dc0094cde908f42dd73"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

DRY_RUN=0; SKIP_TESTS=0; WITH_MAPS=0; NO_APT=0; NO_PIN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --with-maps) WITH_MAPS=1 ;;
    --no-apt) NO_APT=1 ;;
    --no-pin) NO_PIN=1 ;;
    -h|--help) sed -n "2,27p" "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
REPORT="$ROOT/pi_install_report.txt"
WARNINGS=()

say()  { printf '\n=== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; WARNINGS+=("$*"); }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
run()  {
  printf '  $ %s\n' "$*"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

# ---- 1. guards ---------------------------------------------------------------
say "1/7 Checking this machine"
[ -f "$ROOT/pyproject.toml" ] && [ -d "$ROOT/avinavi" ] \
  || die "Cannot find the project next to this script ($ROOT)."
if [ "$(id -u)" -eq 0 ]; then
  die "Do not run with sudo. Run as your normal user; it asks for sudo itself once."
fi

ARCH="$(uname -m)"
MODEL="$( { tr -d '\0' < /proc/device-tree/model; } 2>/dev/null || echo unknown)"
echo "  machine: $MODEL ($ARCH)"
if [ "$ARCH" != "aarch64" ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    warn "not an aarch64 machine ($ARCH) - continuing only because this is a dry run"
  else
    die "This is $ARCH, not a 64-bit Raspberry Pi. Install 64-bit Raspberry Pi OS."
  fi
fi
if [ "$DRY_RUN" -eq 0 ]; then
  USERLAND="$(getconf LONG_BIT)"
  [ "$USERLAND" = "64" ] || die "32-bit userland. Install the 64-bit Raspberry Pi OS image."
  case "$MODEL" in
    *"Raspberry Pi 5"*) ;;
    *) warn "this is not a Raspberry Pi 5 ($MODEL); timings will not represent the target" ;;
  esac
fi

command -v python3 >/dev/null || die "python3 not found."
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
  || die "Python $PYV is too old; need 3.9 or newer."
echo "  python3: $PYV"

FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1048576}')"
echo "  free disk: ${FREE_GB} GB"
[ "$FREE_GB" -ge 4 ] || warn "only ${FREE_GB} GB free; the install needs about 3 GB"

# ---- 2. system packages (the only sudo step) ---------------------------------
say "2/7 System packages"
APT_PKGS=(git python3-venv python3-dev libgl1 libglib2.0-0 v4l-utils)
if [ "$NO_APT" -eq 1 ]; then
  echo "  skipped (--no-apt): ${APT_PKGS[*]}"
else
  run sudo apt-get update
  run sudo apt-get install -y "${APT_PKGS[@]}"
fi

# ---- 3. virtual environment --------------------------------------------------
say "3/7 Python environment in $VENV"
if [ -x "$PY" ]; then
  echo "  already exists, reusing it"
else
  run python3 -m venv "$VENV"
fi
run "$PY" -m pip install --upgrade pip wheel setuptools

# ---- 4. torch (pinned to the laptop's version) ---------------------------------
say "4/7 torch / torchvision (CPU build for aarch64)"
if [ "$NO_PIN" -eq 1 ]; then
  run "$PY" -m pip install --index-url "$TORCH_INDEX" torch torchvision
else
  if ! run "$PY" -m pip install --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"; then
    warn "torch $TORCH_VERSION has no wheel for this Pi; installing the newest instead"
    run "$PY" -m pip install --index-url "$TORCH_INDEX" torch torchvision
  fi
fi

# ---- 5. AviNavi + LightGlue ----------------------------------------------------
say "5/7 AviNavi, LightGlue ($LIGHTGLUE_REF), OpenCV"
run "$PY" -m pip install "lightglue @ git+https://github.com/cvg/LightGlue.git@$LIGHTGLUE_REF"
# Not the "vision" extra: it names LightGlue without a commit and could replace
# the pinned one. Install its other parts explicitly instead.
run "$PY" -m pip install -e "$ROOT" "opencv-python-headless>=4.8" "pytest>=7.0"
if [ "$WITH_MAPS" -eq 1 ]; then
  run "$PY" -m pip install "rasterio>=1.3" "pyproj>=3.5"
fi

# ---- 6. model weights (the aircraft has no network) -----------------------------
say "6/7 Downloading the SuperPoint and LightGlue weights now"
WARMUP='
import time, numpy as np, torch, cv2
from avinavi.config import load_config
from avinavi.localization.engine import LocalizationEngine
cfg = load_config()
ext = LocalizationEngine._build_extractor(cfg)
mat = LocalizationEngine._build_matcher(cfg)
rng = np.random.default_rng(0)
img = cv2.GaussianBlur((rng.random((768, 768)) * 255).astype(np.uint8), (0, 0), 2.0)
f1 = ext.extract(img); f2 = ext.extract(img)
mat.match(f1, f2)
t = time.perf_counter(); ext.extract(img); t_sp = time.perf_counter() - t
t = time.perf_counter(); mat.match(f1, f2); t_lg = time.perf_counter() - t
print("weights OK; torch threads %d" % torch.get_num_threads())
print("SMOKE TIMING (synthetic image, NOT a measurement): superpoint %.0f ms, lightglue %.0f ms/tile"
      % (t_sp * 1000, t_lg * 1000))
'
SMOKE="(not run)"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  $ $PY -c <build extractor + matcher, run once on a synthetic image>"
else
  SMOKE="$(cd "$ROOT" && "$PY" -c "$WARMUP")" || die "Building the models failed (see above)."
  echo "$SMOKE"
  ls "$HOME/.cache/torch/hub/checkpoints/" 2>/dev/null | grep -q superpoint \
    || warn "SuperPoint weights not found in ~/.cache/torch/hub/checkpoints"
fi

# ---- 7. tests + report -----------------------------------------------------------
say "7/7 Tests and report"
TESTS="(skipped)"
if [ "$SKIP_TESTS" -eq 0 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  $ $PY -m pytest -q"
  else
    set +e
    TESTS="$(cd "$ROOT" && "$PY" -m pytest -q 2>&1 | tail -n 3)"
    set -e
    echo "$TESTS"
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run finished - nothing was changed."
  exit 0
fi

# Report only. None of these are changed by this script.
GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
TEMP="$(vcgencmd measure_temp 2>/dev/null || echo unknown)"
THROT="$(vcgencmd get_throttled 2>/dev/null || echo unknown)"
case "$THROT" in
  *=0x0|unknown) ;;
  *) warn "the Pi reports throttling or under-voltage ($THROT): check the power supply and cooling" ;;
esac
id -nG | grep -qw dialout || warn "user $(id -un) is not in the dialout group (needed for the serial link to the flight controller)"
[ -e /dev/ttyAMA0 ] || warn "/dev/ttyAMA0 does not exist (UART to the flight controller is not enabled yet)"

{
  echo "AviNavi Pi install report - $(date -Is)"
  echo "model:      $MODEL"
  echo "os:         $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "kernel:     $(uname -r) ($ARCH)"
  echo "ram:        $(free -m | awk '/^Mem:/ {print $2 " MB"}')"
  echo "disk free:  ${FREE_GB} GB"
  echo "cpu:        $(nproc) cores, governor $GOV"
  echo "temp:       $TEMP"
  echo "throttled:  $THROT"
  echo "git:        $(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
  echo "python:     $("$PY" -V 2>&1)"
  echo "packages:"
  "$PY" -m pip freeze | grep -iE '^(torch|torchvision|opencv|lightglue|numpy|kornia|pymavlink|rasterio)' | sed 's/^/  /'
  echo "models:"
  echo "$SMOKE" | sed 's/^/  /'
  echo "tests:"
  echo "$TESTS" | sed 's/^/  /'
  echo "warnings:"
  if [ "${#WARNINGS[@]}" -eq 0 ]; then echo "  none"; else printf '  %s\n' "${WARNINGS[@]}"; fi
} > "$REPORT"

say "Done. Report: $REPORT"
cat "$REPORT"
