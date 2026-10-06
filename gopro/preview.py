from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

PLACEHOLDER_SIZE = (640, 360)


def _ffmpeg_hwaccel_args() -> list[str]:
    if sys.platform != "darwin":
        return []
    if os.environ.get("GOPRO_PREVIEW_HWACCEL", "1") == "0":
        return []
    return ["-hwaccel", "videotoolbox"]


def resolve_vision_detector_path(explicit_path: Optional[str] = None) -> str:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.environ.get("VISION_DETECTOR_BIN")
    if env_path:
        candidates.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "vision-detector" / ".build" / "release" / "VisionDetector",
            repo_root / "vision-detector" / ".build" / "debug" / "VisionDetector",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    which_path = shutil.which("vision-detector") or shutil.which("VisionDetector")
    if which_path:
        return which_path

    raise FileNotFoundError(
        "vision-detector binary not found; build with `swift build -c release -C vision-detector` "
        "or set VISION_DETECTOR_BIN."
    )


class FrameMailbox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._index = 0
        self._timestamp_ns = 0
        self._consumed_index = 0
        self._drops = 0

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._index != self._consumed_index:
                self._drops += 1
            self._index += 1
            self._frame = frame
            self._timestamp_ns = time.monotonic_ns()

    def consume(self) -> Optional[tuple[np.ndarray, int, int, int]]:
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame
            index = self._index
            timestamp_ns = self._timestamp_ns
            drops = self._drops
            self._consumed_index = index
        return frame, index, timestamp_ns, drops


def _read_latest(cap: cv2.VideoCapture, max_grabs: int = 10) -> tuple[bool, Optional[np.ndarray]]:
    grabbed = False
    for _ in range(max_grabs):
        if not cap.grab():
            break
        grabbed = True
    if grabbed:
        ok, frame = cap.retrieve()
        return ok, frame
    ok, frame = cap.read()
    return ok, frame


@dataclass
class PreviewStats:
    label: str
    backend: str
    last_log: float = 0.0
    frame_count: int = 0

    def update(self, age_ms: float, drops: int) -> None:
        self.frame_count += 1
        now = time.monotonic()
        if self.last_log <= 0:
            self.last_log = now
            return
        if now - self.last_log < 2.0:
            return
        fps = self.frame_count / max(0.01, now - self.last_log)
        print(
            f"preview camera={self.label} fps={fps:.1f} drops={drops} age_ms={age_ms:.0f} backend={self.backend}",
            flush=True,
        )
        self.last_log = now
        self.frame_count = 0


@dataclass
class PoseOutput:
    camera_id: str
    label: str
    overlay_path: Optional[Path]
    landmarks_path: Optional[Path]
    stats_path: Optional[Path]
    record_drops: int = 0


class PreviewReceiver:
    def __init__(
        self,
        port: int,
        *,
        backend: str = "opencv",
        target_size: Optional[tuple[int, int]] = None,
        label: str = "-",
        fps_target: int = 30,
        max_grabs: int = 5,
    ) -> None:
        self.port = port
        self.backend = backend
        self.target_size = target_size
        self.label = label
        self.fps_target = fps_target
        self.max_grabs = max_grabs
        self.mailbox = FrameMailbox()
        self.ready = threading.Event()
        self.started_at: Optional[float] = None
        self._first_frame_logged = False
        self._stop = threading.Event()
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._thread:
            return
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"preview-{self.label}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def mark_stopping(self) -> None:
        self._stopping = True

    def _run(self) -> None:
        if self.backend == "ffmpeg":
            self._run_ffmpeg()
            return
        self._run_opencv()

    def _run_opencv(self) -> None:
        url = f"udp://@0.0.0.0:{self.port}?fifo_size=1000000&overrun_nonfatal=1"
        cap: Optional[cv2.VideoCapture] = None
        warned = False
        try:
            while not self._stop.is_set():
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        if not warned:
                            print(
                                f"preview camera={self.label} failed to open UDP {self.port}",
                                flush=True,
                            )
                            warned = True
                        time.sleep(0.25)
                        continue
                    self.ready.set()
                ok, frame = _read_latest(cap, max_grabs=self.max_grabs)
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                self.mailbox.publish(frame)
                self._log_first_frame()
        finally:
            if cap is not None:
                cap.release()

    def _run_ffmpeg(self) -> None:
        if not self.target_size:
            return
        width, height = self.target_size
        url = f"udp://@0.0.0.0:{self.port}?fifo_size=1000000&overrun_nonfatal=1"
        hwaccel_args = _ffmpeg_hwaccel_args()
        frame_bytes = int(width * height * 3)
        backoff = 0.25
        while not self._stop.is_set():
            self.started_at = time.monotonic()
            self._first_frame_logged = False
            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-analyzeduration",
                "0",
                "-probesize",
                "32768",
                *hwaccel_args,
                "-vsync",
                "0",
                "-i",
                url,
                "-vf",
                f"scale={width}:{height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]
            stderr_tail: deque[str] = deque(maxlen=8)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._proc = proc
            if not proc.stdout:
                return

            def _drain_stderr() -> None:
                if not proc.stderr:
                    return
                while not self._stop.is_set():
                    line = proc.stderr.readline()
                    if not line:
                        break
                    try:
                        text = line.decode(errors="ignore").rstrip()
                    except Exception:
                        text = ""
                    if text:
                        stderr_tail.append(text)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            self.ready.set()
            buffer = bytearray()
            while not self._stop.is_set():
                if proc.poll() is not None:
                    break
                needed = frame_bytes - len(buffer)
                chunk = proc.stdout.read(needed)
                if not chunk:
                    time.sleep(0.01)
                    continue
                buffer.extend(chunk)
                if len(buffer) < frame_bytes:
                    continue
                frame_bytes_buf = bytes(buffer)
                buffer.clear()
                frame = np.frombuffer(frame_bytes_buf, dtype=np.uint8).reshape((height, width, 3))
                self.mailbox.publish(frame)
                self._log_first_frame()

            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

            if self._stop.is_set() or self._stopping:
                break

            if proc.returncode is not None:
                if stderr_tail:
                    detail = " stderr=" + " | ".join(stderr_tail)
                else:
                    detail = ""
                print(
                    f"preview camera={self.label} ffmpeg exited ({proc.returncode}){detail}",
                    flush=True,
                )

            time.sleep(backoff)
            backoff = min(1.0, backoff + 0.25)

        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _log_first_frame(self) -> None:
        if self._first_frame_logged:
            return
        if self.started_at is None:
            return
        self._first_frame_logged = True
        elapsed_ms = (time.monotonic() - self.started_at) * 1000.0
        print(
            f"preview camera={self.label} first_frame_ms={elapsed_ms:.0f} backend={self.backend}",
            flush=True,
        )


class PosePipeline:
    def __init__(
        self,
        *,
        port: int,
        width: int,
        height: int,
        camera_id: str,
        label: str,
        vision_bin: str,
        landmarks_path: Optional[Path],
        stats_path: Optional[Path],
        overlay_path: Optional[Path],
        stats_overlay: bool = True,
        pose_overlay: bool = True,  # Controls --no-overlay flag for skeleton rendering
        async_detection: bool = True,
        metal: bool = True,
        record_overlay: bool = True,
        target_fps: int = 30,
    ) -> None:
        self.port = port
        self.width = width
        self.height = height
        self.camera_id = camera_id
        self.label = label
        self.vision_bin = vision_bin
        self.landmarks_path = landmarks_path
        self.stats_path = stats_path
        self.overlay_path = overlay_path
        self.stats_overlay = stats_overlay
        self.pose_overlay = pose_overlay
        self.async_detection = async_detection
        self.metal = metal
        self.record_overlay = record_overlay
        self.target_fps = target_fps
        self.mailbox = FrameMailbox()
        self.ready = threading.Event()
        self.started_at: Optional[float] = None
        self._first_frame_logged = False
        self._stop = threading.Event()
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self._decoder_proc: Optional[subprocess.Popen] = None
        self._vision_proc: Optional[subprocess.Popen] = None
        self._encoder_proc: Optional[subprocess.Popen] = None
        self._record_queue: deque[bytes] = deque(maxlen=8)
        self._record_drops = 0
        self._record_thread: Optional[threading.Thread] = None

    @property
    def output(self) -> PoseOutput:
        return PoseOutput(
            camera_id=self.camera_id,
            label=self.label,
            overlay_path=self.overlay_path if self.record_overlay else None,
            landmarks_path=self.landmarks_path,
            stats_path=self.stats_path,
            record_drops=self._record_drops,
        )

    def start(self) -> None:
        if self._thread:
            return
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"pose-{self.label}", daemon=True)
        self._thread.start()

    def mark_stopping(self) -> None:
        self._stopping = True

    def stop(self) -> None:
        self._stopping = True
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        if self._encoder_proc and self._encoder_proc.stdin:
            try:
                self._encoder_proc.stdin.close()
            except Exception:
                pass
        self._terminate_proc(self._encoder_proc)
        self._terminate_proc(self._vision_proc)
        self._terminate_proc(self._decoder_proc)
        self._encoder_proc = None
        self._vision_proc = None
        self._decoder_proc = None

    def _terminate_proc(self, proc: Optional[subprocess.Popen]) -> None:
        if not proc or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _build_decoder_cmd(self) -> list[str]:
        url = f"udp://@0.0.0.0:{self.port}?fifo_size=200000&overrun_nonfatal=1"
        return [
            "ffmpeg",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-analyzeduration",
            "0",
            "-probesize",
            "32768",
            "-i",
            url,
            "-vf",
            f"fps={max(5, int(self.target_fps))}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-s",
            f"{self.width}x{self.height}",
            "-vsync",
            "0",
            "-",
        ]

    def _build_vision_cmd(self) -> list[str]:
        cmd = [
            self.vision_bin,
            "transform",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--pixel-format",
            "bgra",
            "--fps",
            str(self.target_fps),
            "--camera",
            self.camera_id,
            "--stats",
        ]
        if not self.stats_overlay:
            cmd.append("--no-stats-overlay")
        if not self.pose_overlay:
            cmd.append("--no-overlay")
        if self.async_detection:
            cmd.append("--async")
        if self.metal:
            cmd.append("--metal")
        if self.landmarks_path:
            cmd.extend(["--landmarks-out", str(self.landmarks_path)])
        if self.stats_path:
            cmd.extend(["--stats-out", str(self.stats_path)])
        return cmd

    def _build_encoder_cmd(self, path: Path) -> list[str]:
        video_encoder = "libx264"
        extra: list[str] = ["-preset", "veryfast", "-tune", "zerolatency"]
        if sys.platform == "darwin" and os.environ.get("GOPRO_PREVIEW_HWACCEL", "1") != "0":
            video_encoder = "h264_videotoolbox"
            extra = []
        return [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.target_fps),
            "-i",
            "-",
            "-c:v",
            video_encoder,
            *extra,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ]

    def _start_encoder(self) -> None:
        if not self.overlay_path or not self.record_overlay:
            return
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._build_encoder_cmd(self.overlay_path)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if not proc.stdin:
            return
        self._encoder_proc = proc
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

    def _record_loop(self) -> None:
        if not self._encoder_proc or not self._encoder_proc.stdin:
            return
        while not self._stop.is_set():
            if not self._record_queue:
                time.sleep(0.005)
                continue
            frame = self._record_queue.popleft()
            try:
                self._encoder_proc.stdin.write(frame)
            except Exception:
                break

    def _enqueue_record(self, frame: bytes) -> None:
        if not self.record_overlay or not self._encoder_proc:
            return
        if len(self._record_queue) >= self._record_queue.maxlen:
            self._record_queue.popleft()
            self._record_drops += 1
        self._record_queue.append(frame)

    def _log_first_frame(self) -> None:
        if self._first_frame_logged:
            return
        if self.started_at is None:
            return
        self._first_frame_logged = True
        elapsed_ms = (time.monotonic() - self.started_at) * 1000.0
        print(
            f"preview camera={self.label} first_frame_ms={elapsed_ms:.0f} backend=pose",
            flush=True,
        )

    def _run(self) -> None:
        decode_cmd = self._build_decoder_cmd()
        decoder = subprocess.Popen(
            decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if not decoder.stdout:
            return
        self._decoder_proc = decoder

        vision_cmd = self._build_vision_cmd()
        vision = subprocess.Popen(
            vision_cmd,
            stdin=decoder.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        decoder.stdout.close()
        if not vision.stdout:
            return
        self._vision_proc = vision
        self.ready.set()
        self._start_encoder()

        frame_bytes = int(self.width * self.height * 4)
        buffer = bytearray()
        while not self._stop.is_set():
            if vision.poll() is not None:
                break
            needed = frame_bytes - len(buffer)
            chunk = vision.stdout.read(needed)
            if not chunk:
                time.sleep(0.01)
                continue
            buffer.extend(chunk)
            if len(buffer) < frame_bytes:
                continue
            frame_bytes_buf = bytes(buffer)
            buffer.clear()
            frame = np.frombuffer(frame_bytes_buf, dtype=np.uint8).reshape((self.height, self.width, 4))
            bgr = frame[:, :, :3].copy()
            self.mailbox.publish(bgr)
            self._enqueue_record(frame_bytes_buf)
            self._log_first_frame()

        self._terminate_proc(vision)
        self._terminate_proc(decoder)
        self._terminate_proc(self._encoder_proc)


def _overlay_info(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(
        frame,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def run_preview(
    ports: list[int],
    labels: list[str],
    *,
    backend: str = "opencv",
    size: Optional[tuple[int, int]] = None,
    on_start: Optional[Callable[[], None]] = None,
) -> None:
    receivers = [
        PreviewReceiver(port, backend=backend, target_size=size, label=label)
        for port, label in zip(ports, labels)
    ]
    for receiver in receivers:
        receiver.start()
    ready_deadline = time.monotonic() + 1.5
    for receiver in receivers:
        timeout = max(0.0, ready_deadline - time.monotonic())
        if timeout <= 0:
            break
        if not receiver.ready.wait(timeout=timeout):
            print(f"preview camera={receiver.label} receiver not ready before stream start", flush=True)
    if on_start:
        on_start()

    last_frames: list[Optional[np.ndarray]] = [None for _ in receivers]
    stats = [PreviewStats(label, backend) for label in labels]
    placeholder_size = size or PLACEHOLDER_SIZE
    cv2.namedWindow("GoPro Preview", cv2.WINDOW_NORMAL)

    try:
        while True:
            now_ns = time.monotonic_ns()
            for idx, receiver in enumerate(receivers):
                payload = receiver.mailbox.consume()
                if payload:
                    frame, frame_index, ts_ns, drops = payload
                    age_ms = (now_ns - ts_ns) / 1e6
                    text = f"{labels[idx]} :{ports[idx]} #{frame_index} {age_ms:.0f}ms"
                    frame = _overlay_info(frame.copy(), text)
                    last_frames[idx] = frame
                    stats[idx].update(age_ms, drops)

            if len(receivers) == 1:
                frame = last_frames[0]
                if frame is None:
                    placeholder = np.zeros((placeholder_size[1], placeholder_size[0], 3), dtype=np.uint8)
                    _overlay_info(placeholder, f"{labels[0]} waiting for stream")
                    cv2.imshow("GoPro Preview", placeholder)
                else:
                    cv2.imshow("GoPro Preview", frame)
            else:
                frames = [f for f in last_frames if f is not None]
                cell_w, cell_h = placeholder_size
                if frames:
                    cell_h, cell_w = frames[0].shape[:2]
                cols = 2
                rows = (len(receivers) + cols - 1) // cols
                grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
                for idx, frame in enumerate(last_frames):
                    if frame is None:
                        placeholder = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                        _overlay_info(placeholder, f"{labels[idx]} waiting")
                        frame = placeholder
                    if frame.shape[0] != cell_h or frame.shape[1] != cell_w:
                        frame = cv2.resize(frame, (cell_w, cell_h))
                    r = idx // cols
                    c = idx % cols
                    grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = frame
                cv2.imshow("GoPro Preview", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                for receiver in receivers:
                    receiver.mark_stopping()
                break
    except KeyboardInterrupt:
        for receiver in receivers:
            receiver.mark_stopping()
    finally:
        for receiver in receivers:
            receiver.stop()
        cv2.destroyAllWindows()


def run_pose_preview(
    ports: list[int],
    labels: list[str],
    camera_ids: list[str],
    *,
    size: tuple[int, int],
    vision_bin: str,
    landmarks_paths: list[Optional[Path]],
    stats_paths: list[Optional[Path]],
    overlay_paths: list[Optional[Path]],
    stats_overlay: bool = True,
    async_detection: bool = True,
    metal: bool = True,
    target_fps: int = 30,
    on_start: Optional[Callable[[], None]] = None,
) -> list[PoseOutput]:
    width, height = size
    pipelines: list[PosePipeline] = []
    for port, label, camera_id, landmarks_path, stats_path, overlay_path in zip(
        ports,
        labels,
        camera_ids,
        landmarks_paths,
        stats_paths,
        overlay_paths,
    ):
        pipeline = PosePipeline(
            port=port,
            width=width,
            height=height,
            camera_id=camera_id,
            label=label,
            vision_bin=vision_bin,
            landmarks_path=landmarks_path,
            stats_path=stats_path,
            overlay_path=overlay_path,
            stats_overlay=stats_overlay,
            async_detection=async_detection,
            metal=metal,
            record_overlay=overlay_path is not None,
            target_fps=target_fps,
        )
        pipelines.append(pipeline)

    for pipeline in pipelines:
        pipeline.start()
    ready_deadline = time.monotonic() + 1.5
    for pipeline in pipelines:
        timeout = max(0.0, ready_deadline - time.monotonic())
        if timeout <= 0:
            break
        if not pipeline.ready.wait(timeout=timeout):
            print(f"preview camera={pipeline.label} receiver not ready before stream start", flush=True)

    if on_start:
        on_start()

    last_frames: list[Optional[np.ndarray]] = [None for _ in pipelines]
    stats = [PreviewStats(label, "pose") for label in labels]
    placeholder_size = size or PLACEHOLDER_SIZE
    cv2.namedWindow("GoPro Preview", cv2.WINDOW_NORMAL)

    try:
        while True:
            now_ns = time.monotonic_ns()
            for idx, pipeline in enumerate(pipelines):
                payload = pipeline.mailbox.consume()
                if payload:
                    frame, frame_index, ts_ns, drops = payload
                    age_ms = (now_ns - ts_ns) / 1e6
                    text = f"{labels[idx]} :{ports[idx]} #{frame_index} {age_ms:.0f}ms"
                    frame = _overlay_info(frame.copy(), text)
                    last_frames[idx] = frame
                    stats[idx].update(age_ms, drops)

            if len(pipelines) == 1:
                frame = last_frames[0]
                if frame is None:
                    placeholder = np.zeros((placeholder_size[1], placeholder_size[0], 3), dtype=np.uint8)
                    _overlay_info(placeholder, f"{labels[0]} waiting for stream")
                    cv2.imshow("GoPro Preview", placeholder)
                else:
                    cv2.imshow("GoPro Preview", frame)
            else:
                frames = [f for f in last_frames if f is not None]
                cell_w, cell_h = placeholder_size
                if frames:
                    cell_h, cell_w = frames[0].shape[:2]
                cols = 2
                rows = (len(pipelines) + cols - 1) // cols
                grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
                for idx, frame in enumerate(last_frames):
                    if frame is None:
                        placeholder = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                        _overlay_info(placeholder, f"{labels[idx]} waiting")
                        frame = placeholder
                    if frame.shape[0] != cell_h or frame.shape[1] != cell_w:
                        frame = cv2.resize(frame, (cell_w, cell_h))
                    r = idx // cols
                    c = idx % cols
                    grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = frame
                cv2.imshow("GoPro Preview", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                for pipeline in pipelines:
                    pipeline.mark_stopping()
                break
    except KeyboardInterrupt:
        for pipeline in pipelines:
            pipeline.mark_stopping()
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

    return [pipeline.output for pipeline in pipelines]
