"""
main.py — Entry point for the FRDM-IMX95 Edge AI Vision application.

Starts the inference loop and the MJPEG streaming server.
Open http://<board_ip>:5000 in your browser to see the live annotated feed.

Usage:
    python3 main.py [--config config.json] [--no-npu] [--verbose]
"""

import argparse
import json
import logging
import random
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np

import actions
import inference
import overlay
import streamer

# EMA smoothing factor for all runtime statistics (FPS, latencies).
# α=0.01 ≈ 100-frame window; lower = smoother but slower to react.
_STATS_ALPHA = 0.01


def _list_cameras() -> list[dict]:
    """Return [{device, name}] for USB-connected V4L2 video capture devices.

    v4l2-ctl --list-devices marks USB sources with '(usb-...)' in the header
    line and platform/ISP sources with '(platform:...)'.  Only the first
    /dev/video* entry per USB camera is returned (subsequent entries are
    metadata nodes, not capture devices).
    """
    import subprocess
    cameras: list[dict] = []
    seen_names: set[str] = set()
    try:
        out = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=3
        ).stdout
        current_name: str | None = None
        is_usb: bool = False
        for line in out.splitlines():
            stripped = line.strip()
            if not stripped:
                current_name = None
                is_usb = False
            elif not line[0].isspace():
                current_name = stripped.rstrip(':')
                is_usb = 'usb' in current_name.lower()
            elif (is_usb
                  and stripped.startswith('/dev/video')
                  and current_name not in seen_names):
                seen_names.add(current_name)
                cameras.append({'device': stripped, 'name': current_name})
    except Exception:
        pass
    return cameras


def resolve_model_path(config: dict) -> str:
    """
    Derive the pipeline.json path from models_dir + variant.
    Sets config["model"]["path"] in-place so load_pipeline() can use it.
    """
    m = config["model"]
    source = m["variants"][m["variant"]]
    path = str(Path(m["models_dir"]) / source / "pipeline.json")
    m["path"] = path
    return path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the JSON config file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def _open_capture(cam_cfg: dict) -> tuple[cv2.VideoCapture, str]:
    """
    Open a cv2.VideoCapture from either a local device or a remote MJPEG URL.

    Returns:
        (cap, source_description)
    """
    source = cam_cfg.get("source", "local")

    if source == "remote":
        url = cam_cfg.get("remote_url", "http://192.168.7.1:5001/stream")
        logger.info(f"Remote camera mode — connecting to {url}")
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"Could not open remote stream: {url}")
            logger.error("Ensure host/webcam_streamer.py is running on the laptop.")
            sys.exit(1)
        logger.info(f"Remote stream connected: {url}")
        return cap, f"remote:{url}"

    # Local camera (default)
    device = cam_cfg.get("device", "/dev/video0")
    logger.info(f"Opening local camera {device} at "
                f"{cam_cfg['width']}x{cam_cfg['height']} @ {cam_cfg['fps']} fps")
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS,          cam_cfg["fps"])
    if not cap.isOpened():
        logger.error(f"Could not open camera at {device}")
        sys.exit(1)
    return cap, f"local:{device}"


def _capture_loop(cap: cv2.VideoCapture, frame_queue: Queue,
                  stop_event: threading.Event) -> None:
    """
    Background thread: continuously read frames from cap and put them into
    frame_queue. Uses drop-oldest semantics (maxsize=2) so the inference
    thread always processes the most recent frame, never a stale one.
    """
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame read failed — retrying...")
            time.sleep(0.05)
            continue
        # Drop oldest frame if the inference thread hasn't consumed it yet
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put_nowait(frame)
    cap.release()
    logger.debug("Capture thread stopped.")


def _preprocess_loop(frame_queue: Queue, preproc_queue: Queue,
                     input_details: list, stop_event: threading.Event) -> None:
    """
    Background thread: preprocess frames and push
    (frame, input_tensor, 0.0) into preproc_queue.
    The 0.0 is the invoke_ms accumulator, reset at each frame.
    """
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue
        input_data = inference.preprocess_frame(frame, input_details)
        if preproc_queue.full():
            try:
                preproc_queue.get_nowait()
            except Empty:
                pass
        preproc_queue.put_nowait((frame, [input_data], 0.0))
    logger.debug("Preprocess thread stopped.")


def _open_gst_pipeline(cam_cfg: dict):
    """
    Build a GStreamer pipeline that reads the configured source and uses
    imxvideoconvert_g2d to convert to RGB in the G2D engine.

    No output resolution is forced — the camera negotiates its native resolution
    so the aspect ratio is preserved.  _gst_capture_loop resizes to model input
    size for the inference tensor; the full native frame is used for display.

    Returns (Gst.Pipeline, appsink) on success, or (None, None) if GStreamer
    or imxvideoconvert_g2d is unavailable — the caller falls back to OpenCV.
    """
    if not cam_cfg.get("use_gstreamer", False):
        return None, None

    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
    except Exception as e:
        logger.info(f"GStreamer not available ({e}), using OpenCV.")
        return None, None

    # Fast check before building the pipeline — avoids an opaque FAILURE later.
    if Gst.ElementFactory.find('imxvideoconvert_g2d') is None:
        logger.info("imxvideoconvert_g2d not found in GStreamer registry, using OpenCV.")
        return None, None

    source = cam_cfg.get("source", "local")
    if source == "remote":
        url = cam_cfg.get("remote_url", "http://192.168.7.1:5001/stream")
        src = f'souphttpsrc location="{url}" is-live=true ! multipartdemux ! jpegdec'
    else:
        device = cam_cfg.get("device", "/dev/video0")
        src = f'v4l2src device={device}'

    pipe_str = (
        f'{src} ! imxvideoconvert_g2d '
        f'! video/x-raw,format=RGB '
        f'! appsink name=sink sync=false max-buffers=2 drop=true'
    )

    pipeline = None
    try:
        pipeline = Gst.parse_launch(pipe_str)
        appsink  = pipeline.get_by_name('sink')
        pipeline.set_state(Gst.State.PLAYING)
        ret, _, _ = pipeline.get_state(2 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            bus = pipeline.get_bus()
            msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR)
            detail = ""
            if msg:
                err, debug = msg.parse_error()
                detail = f": {err.message} — {debug}"
            raise RuntimeError(f"state change FAILURE{detail}")
        logger.info("GStreamer G2D pipeline started — native camera resolution, RGB")
        return pipeline, appsink
    except Exception as e:
        logger.warning(f"GStreamer pipeline failed ({e}), falling back to OpenCV.")
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        return None, None


def _gst_capture_loop(appsink, preproc_queue: Queue,
                      input_h: int, input_w: int,
                      input_dtype, stop_event: threading.Event) -> None:
    """
    Pull RGB frames from the GStreamer appsink, build the int8/float input tensor,
    and push (bgr_frame, [input_tensor], 0.0) directly to the first stage queue.

    G2D outputs at display resolution (cam width × height).  If that differs from
    the model input size we do a cheap CPU resize for the tensor only — the full-
    resolution bgr frame is kept for annotation and MJPEG streaming so the image
    matches what the OpenCV path produces.
    """
    import gi
    from gi.repository import Gst

    while not stop_event.is_set():
        sample = appsink.emit('try-pull-sample', 100 * Gst.MSECOND)
        if sample is None:
            continue

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        s    = caps.get_structure(0)
        h, w = s.get_value('height'), s.get_value('width')

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            continue
        rgb = np.ndarray((h, w, 3), dtype=np.uint8, buffer=map_info.data).copy()
        buf.unmap(map_info)

        # Full-resolution BGR frame for annotation and streaming
        bgr = rgb[:, :, ::-1].copy()

        # Resize to model input size only if display resolution differs
        if h != input_h or w != input_w:
            rgb_small = cv2.resize(rgb, (input_w, input_h))
        else:
            rgb_small = rgb

        # Build inference input tensor
        if input_dtype == np.int8:
            tensor = np.expand_dims(rgb_small.astype(np.int16) - 128, axis=0).astype(np.int8)
        elif input_dtype == np.uint8:
            tensor = np.expand_dims(rgb_small, axis=0)
        else:
            tensor = np.expand_dims(rgb_small / 255.0, axis=0).astype(np.float32)

        if preproc_queue.full():
            try:
                preproc_queue.get_nowait()
            except Empty:
                pass
        preproc_queue.put_nowait((bgr, [tensor], 0.0))
    logger.debug("GStreamer capture thread stopped.")


def _stage_loop(stage: dict, in_queue: Queue, out_queue: Queue,
                stop_event: threading.Event) -> None:
    """
    Generic pipeline stage thread.

    Reads  (frame, tensor, invoke_ms_acc) from in_queue.
    Runs   inference.invoke_stage(stage, tensor).
    Pushes (frame, output_tensor, invoke_ms_acc + stage_invoke_ms) to out_queue.

    Works for any number of stages (CPU pre, NPU, CPU post, or any combination).
    invoke_ms accumulates only NPU stage time (invoke_stage returns 0.0 for CPU stages).
    """
    label = stage["label"]
    while not stop_event.is_set():
        try:
            frame, data, invoke_ms_acc = in_queue.get(timeout=1.0)
        except Empty:
            continue
        out_data, elapsed_ms = inference.invoke_stage(stage, data)
        stage["avg_ms"] += _STATS_ALPHA * (elapsed_ms - stage["avg_ms"])
        npu_ms = elapsed_ms if stage["is_npu"] else 0.0
        item = (frame, out_data, invoke_ms_acc + npu_ms)
        if out_queue.full():
            try:
                out_queue.get_nowait()
            except Empty:
                pass
        out_queue.put_nowait(item)
    logger.debug(f"Stage '{label}' thread stopped.")


def run(config: dict) -> None:
    """
    Main loop: capture → inference → annotate → stream.

    Camera source is selected by config["camera"]["source"]:
      "local"  — use /dev/video0 (or configured device) on the board
      "remote" — pull MJPEG stream from laptop webcam_streamer.py

    In both cases a background thread feeds frames into a Queue(maxsize=2)
    so capture and inference run concurrently.

    Args:
        config: Full application config dict
    """
    cam_cfg    = config["camera"]
    stop_event = threading.Event()

    # Auto-detect camera device if none is specified in config.
    # Only USB cameras are considered (platform/ISP devices are excluded).
    # Candidates are shuffled, then probed until one can grab a frame.
    if cam_cfg.get('source', 'local') == 'local' and not cam_cfg.get('device'):
        candidates = _list_cameras()
        if not candidates:
            logger.error("No USB cameras found — connect a USB camera and restart.")
            sys.exit(1)
        random.shuffle(candidates)

        chosen = None
        for cam in candidates:
            probe = cv2.VideoCapture(cam['device'])
            ok = probe.isOpened() and probe.grab()
            probe.release()
            if ok:
                chosen = cam
                break
            logger.debug(f"Skipping {cam['device']} (not a capture device)")

        if chosen is None:
            logger.error("No working USB camera found — connect a USB camera and restart.")
            sys.exit(1)
        cam_cfg['device'] = chosen['device']
        logger.info(f"Auto-selected camera: {chosen['name']} ({chosen['device']})")

    # --- Load pipeline and labels first (input shape needed for GStreamer setup) ---
    logger.info("Loading pipeline...")
    pipeline = inference.load_pipeline(config)
    for stage in pipeline:
        stage["avg_ms"] = 0.0
    labels     = inference.load_labels(config["model"]["labels_path"])
    model_name = Path(config["model"]["path"]).stem
    logger.info(f"Labels loaded: {len(labels)} classes | Pipeline stages: {len(pipeline)}")

    inp_det    = pipeline[0]["input_details"][0]
    input_h, input_w = inp_det['shape'][1], inp_det['shape'][2]
    input_dtype      = inp_det['dtype']

    # --- Camera capture + preprocessing ---
    # Try GStreamer with imxvideoconvert_g2d (resize + RGB conversion in G2D hardware).
    # Fall back to OpenCV + CPU preprocess if unavailable.
    preproc_queue: Queue = Queue(maxsize=2)
    gst_pipe, appsink = _open_gst_pipeline(cam_cfg)

    # Mutable container so _switch_camera can replace the stop event in-place.
    capture_state: dict = {'stop': threading.Event(), 'gst_pipe': gst_pipe}

    def _start_opencv_capture(device: str) -> None:
        """Start OpenCV capture + preprocess threads feeding into preproc_queue."""
        cam_cfg['device'] = device
        cam_cfg['source'] = 'local'
        cap, desc = _open_capture(cam_cfg)
        logger.info(f"Camera source: {desc}")
        fq: Queue = Queue(maxsize=2)
        s = capture_state['stop']
        threading.Thread(target=_capture_loop, args=(cap, fq, s),
                         daemon=True, name='FrameCapture').start()
        threading.Thread(target=_preprocess_loop,
                         args=(fq, preproc_queue, pipeline[0]['input_details'], s),
                         daemon=True, name='Preprocess').start()

    def _switch_camera(new_device: str) -> None:
        """Hot-switch to a different camera; always uses the OpenCV path."""
        logger.info(f"Camera switch requested: {new_device}")
        # Pre-check — avoids stopping current capture if the device is broken.
        probe = cv2.VideoCapture(new_device)
        if not probe.isOpened():
            logger.error(f"Cannot open {new_device} — switch aborted")
            probe.release()
            return
        probe.release()
        # Stop GStreamer pipeline if active.
        gp = capture_state.get('gst_pipe')
        if gp is not None:
            try:
                import gi
                from gi.repository import Gst
                gp.set_state(Gst.State.NULL)
            except Exception:
                pass
            capture_state['gst_pipe'] = None
        # Signal current capture threads to stop, then start fresh.
        capture_state['stop'].set()
        time.sleep(0.2)
        capture_state['stop'] = threading.Event()
        _start_opencv_capture(new_device)
        streamer.set_active_camera(new_device)
        logger.info(f"Camera switched to {new_device}")

    if gst_pipe is not None:
        threading.Thread(
            target=_gst_capture_loop,
            args=(appsink, preproc_queue, input_h, input_w, input_dtype, capture_state['stop']),
            daemon=True, name="GstCapture"
        ).start()
        logger.info("GStreamer G2D capture + preprocess thread started.")
    else:
        _start_opencv_capture(cam_cfg.get('device', '/dev/video0'))
    streamer.set_active_camera(cam_cfg.get('device', ''))
    streamer.register_camera_callbacks(_list_cameras, _switch_camera)

    # --- One stage thread per pipeline stage, chained by queues ---
    # preproc_queue → stage[0] → q1 → stage[1] → q2 → ... → raw_queue
    stage_in_queue = preproc_queue
    for stage in pipeline:
        stage_out_queue: Queue = Queue(maxsize=2)
        threading.Thread(
            target=_stage_loop,
            args=(stage, stage_in_queue, stage_out_queue, stop_event),
            daemon=True,
            name=f"Stage-{stage['label']}"
        ).start()
        logger.info(f"Stage '{stage['label']}' thread started.")
        stage_in_queue = stage_out_queue
    raw_queue = stage_in_queue  # output of last stage

    # --- Start streaming server ---
    streamer.start(config)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        board_ip = s.getsockname()[0]
        s.close()
    except Exception:
        board_ip = "0.0.0.0"
    stream_url = f"http://{board_ip}:{config['streaming']['port']}"
    logger.info(f"Stream available at: {stream_url}")
    logger.info(f"Status endpoint:     {stream_url}/status")

    # --- Stats tracking (EMA, shared alpha) ---
    fps = 0.0
    invoke_ms_avg   = 0.0
    inference_ms_avg = 0.0
    frame_count = 0
    t_start = time.monotonic()

    logger.info("Starting inference loop. Press Ctrl+C to stop.")

    try:
        while True:
            t_frame = time.monotonic()

            # Get output of last pipeline stage (ran concurrently with previous postprocess)
            try:
                frame, raw_list, invoke_ms = raw_queue.get(timeout=1.0)
            except Empty:
                logger.warning("No frame received in 1s — waiting for camera...")
                continue

            # Dequantize last stage output if int8 (invoke_stage does not dequantize)
            last_out_det = pipeline[-1]["output_details"]
            raw = raw_list[0]  # final stage always produces a single detection tensor
            if raw.dtype == np.int8:
                scale, zero_point = last_out_det[0]["quantization"]
                raw = (raw.astype(np.float32) - zero_point) * scale

            # Postprocess (CPU, overlaps with next invoke on the stage threads)
            t_post = time.monotonic()
            if config["inference"].get("skip_postprocess", False):
                detections = []
            else:
                detections = inference.postprocess_detections(raw, frame, config)
            inference_ms = (time.monotonic() - t_post) * 1000

            # Annotate frame (returns a copy — original frame unchanged)
            annotated = overlay.annotate_frame(
                frame, detections, labels, fps, config, model_name, inference_ms, invoke_ms
            )

            # Dispatch action hooks
            actions.on_frame(frame, detections, labels, config)

            # Push annotated frame to the MJPEG streamer
            stage_latency = [
                {"label": s["label"], "avg_ms": round(s["avg_ms"], 1)}
                for s in pipeline
            ]
            streamer.push_frame(annotated, detections, labels, fps,
                                invoke_ms=invoke_ms_avg, inference_ms=inference_ms_avg,
                                stage_latency=stage_latency)

            # Update all stats with shared EMA alpha
            elapsed = time.monotonic() - t_frame
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps              += _STATS_ALPHA * (instant_fps  - fps)
            invoke_ms_avg    += _STATS_ALPHA * (invoke_ms    - invoke_ms_avg)
            inference_ms_avg += _STATS_ALPHA * (inference_ms - inference_ms_avg)

            frame_count += 1
            if frame_count % 100 == 0:
                uptime = time.monotonic() - t_start
                logger.info(f"Frame {frame_count} | FPS: {fps:.1f} | "
                            f"Detections: {len(detections)} | "
                            f"Uptime: {uptime:.0f}s")

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        stop_event.set()
        capture_state['stop'].set()
        gp = capture_state.get('gst_pipe')
        if gp is not None:
            try:
                import gi
                from gi.repository import Gst
                gp.set_state(Gst.State.NULL)
            except Exception:
                pass
        logger.info("Stopped. Goodbye.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRDM-IMX95 Edge AI Vision — inference + MJPEG streamer"
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)"
    )
    parser.add_argument(
        "--no-npu", action="store_true",
        help="Disable NPU delegate, run on CPU only"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    resolve_model_path(config)
    logger.info(f"Model path: {config['model']['path']}")

    if args.no_npu:
        config["model"]["use_npu"] = False
        logger.info("NPU disabled via --no-npu flag")

    # Graceful shutdown on SIGTERM (e.g., systemd stop)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    run(config)


if __name__ == "__main__":
    main()
