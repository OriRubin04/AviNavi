# USB camera on a Raspberry Pi 5

**Summary: a USB camera that works perfectly on the bench will match poorly in
the air for three reasons, and none of them produces an error message.** The
camera opens, frames arrive, the images look fine on screen, and the lock rate
is bad. This page is the checklist.

All three defaults are now set correctly in `config/default.yaml`, so the main
value of this page is knowing what to verify and why each setting matters.

## 1. Pixel format: MJPG, not YUYV

Most UVC cameras default to uncompressed YUYV. At 1280×720, YUYV is 2 bytes per
pixel:

```
1280 × 720 × 2 bytes × 30 fps = 55.3 MB/s = 442 Mbit/s
```

USB 2.0 is nominally 480 Mbit/s and delivers perhaps 320 Mbit/s of isochronous
payload in practice. So the request cannot be met, and the camera does not
refuse it — it quietly serves a lower frame rate, or substitutes a smaller frame
size.

The second failure is worse than the first. **If the camera silently gives you
640×480 when you asked for 1280×720, your `camera.fx/fy/cx/cy` are wrong by a
factor of two**, and every metric quantity downstream — the rectification, the
ground resolution, the altitude reasoning — is wrong with them. AviNavi logs a
warning naming the substituted size and the scale factor, but it cannot know
which of the two you intended.

Check what your camera actually offers:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

Look for `[1]: 'MJPG' (Motion-JPEG, compressed)` and confirm your resolution and
frame rate appear underneath it. Then confirm what you are getting:

```bash
v4l2-ctl --get-fmt-video -d /dev/video0
```

**Ordering matters.** The FourCC must be set before the resolution, because V4L2
validates a frame size against the format currently selected. Setting size first
and format second can leave the camera on a size the new format does not
support, which it resolves by substituting one that is. AviNavi does this in the
right order; if you write your own capture code, do the same.

MJPG costs a JPEG decode per frame. On a Pi 5 that is a few milliseconds and
entirely worth it.

## 2. Exposure: lock it short

This is the one that costs the most lock rate and is the least obvious.

A downward camera over varying terrain — a bright field, then a dark wood, then
water — makes auto-exposure hunt, and when it meters a dark scene it reaches for
a long shutter. Motion blur follows, and **SuperPoint does not detect smeared
corners at all.** It is not that the descriptors degrade; the keypoints are
simply not there.

The arithmetic, for a 120 m cruise with a 900 px focal length:

```
ground resolution = height / focal length = 120 / 900 = 0.133 m/px
```

| Shutter | Ground travel at 22 m/s | Blur in live pixels |
| --- | --- | --- |
| 1/30 s | 0.73 m | 5.5 px |
| 1/60 s | 0.37 m | 2.8 px |
| 1/250 s | 0.09 m | 0.7 px |
| **1/500 s** | **0.04 m** | **0.3 px** |

Anything past about one pixel of blur starts costing keypoints. 1/500 s is the
default (`camera.max_exposure_s: 0.002`).

A short shutter needs light, and the price is gain — which means noise. **Pay
it.** Noise costs far fewer keypoints than blur does: a noisy corner is still a
corner, while a smeared one is not a feature at all.

Verify the lock took:

```bash
v4l2-ctl -d /dev/video0 --get-ctrl=auto_exposure,exposure_time_absolute
```

`auto_exposure` should read `1` (manual; `3` is the usual auto) and
`exposure_time_absolute` should read `20`, in units of 100 µs.

Those units are the trap. V4L2 UVC drivers use 100 µs steps, some OpenCV
backends take seconds, and a few take an opaque index. AviNavi prefers
`v4l2-ctl` when it is installed, because it is the one path with a documented
unit, and otherwise writes through OpenCV, reads the value back, and warns if
the two disagree rather than assuming it worked. Install it:

```bash
sudo apt install -y v4l-utils
```

## 3. Buffer staleness: always take the newest frame

V4L2 commonly queues four buffers and ignores `CAP_PROP_BUFFERSIZE`. A naive
`capture.read()` returns the **oldest** queued frame.

With a 30 fps camera and a localization loop running at 2 Hz, the queue is
always full, so the frame you are handed can be ~0.5 s old. At 22 m/s that is
**eleven metres**, and it does not look like a timing problem — it looks like a
constant position offset, indistinguishable from a mis-georeferenced tile. (That
is not hypothetical: a stale `index.json` produced exactly that signature during
development, and chasing the wrong one of the two costs a day.)

So capture runs continuously on its own thread and the localization loop is
handed the newest frame available, timestamped **at capture**, not at read.
Dropping frames here is the design, not a shortcoming: a 30 fps camera feeding a
2 Hz localizer should discard most of what it captures, and `dropped_frames` in
the diagnostics counting up is the system working. `stale_frames` counting up is
not — that means the capture thread is not keeping up.

```yaml
camera:
  drain_to_newest: true
  max_frame_age_s: 0.25    # reported, not silently localized
```

## Choosing a camera

What matters, roughly in order:

1. **Global shutter**, if you can get it. A rolling shutter shears the image
   under roll rate: at 22 m/s with a 1/30 s readout and 30°/s of roll, the top
   and bottom of the frame are taken from meaningfully different attitudes, and
   a single homography cannot describe that. It is survivable — the residual
   lands in the RANSAC inliers — but a global shutter removes the whole problem.
2. **MJPG at your target resolution and frame rate.** Check with
   `--list-formats-ext` before buying, not after.
3. **Manual exposure and gain controls that actually work.** Some cheap UVC
   cameras accept the control and ignore it. Verify with `v4l2-ctl --get-ctrl`.
4. **A lens whose field of view suits your cruise altitude.** A 60° horizontal
   field at 120 m sees about 139 m across, which at a 0.5 m/px basemap is 277
   tile pixels — ample. A very narrow lens sees too little ground to be
   distinctive; a very wide one wastes resolution on the oblique edges that
   rectification has to throw away anyway.
5. **USB 3.0** if the camera offers it, which removes the bandwidth question
   entirely. The Pi 5 has two USB 3.0 ports — use those, not the 2.0 pair.

## Calibration

The intrinsics are not optional for rectification: without `fx` and `fy` the
geometry is undetermined, `FramePrewarper.available` is `False`, and the system
falls back to matching raw frames at a markedly lower hit rate.

Calibrate at **exactly the resolution you will fly**, and recalibrate if you
change it. A chessboard and `cv2.calibrateCamera` is enough; what matters is
that `fx`, `fy`, `cx`, `cy` and the resolution in `config/default.yaml` all
describe the same configuration.

Then verify the mounting yaw against a real frame, because an error there is a
yaw bias the EKF fights rather than rejects — see `docs/coordinate-frames.md`
for the four common mountings and the `yaw_deg` each one needs.

## Quick bench check

```bash
# What the camera offers, and what it is currently set to
v4l2-ctl --list-formats-ext -d /dev/video0
v4l2-ctl --get-fmt-video -d /dev/video0
v4l2-ctl -d /dev/video0 --get-ctrl=auto_exposure,exposure_time_absolute

# Sustained frame rate at the configured format, which is the number that
# tells you whether the bus is coping
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG \
         --stream-mmap --stream-count=200 --stream-to=/dev/null
```

That last command prints the achieved frame rate. If it is well below what you
asked for, go back to point 1 — nothing further up the stack can recover it.
