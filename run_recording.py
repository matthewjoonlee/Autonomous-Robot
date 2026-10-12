"""Record every camera frame of a run for replay, without slowing control.

Frames are JPEG-encoded on a worker thread and kept in memory; nothing touches
the disk until `save` runs after the motors are stopped. When encoding falls
behind or the byte budget is spent, frames are dropped and counted instead of
delaying the control loop.
"""

import queue
import threading


class FrameRecorder:
    def __init__(self, quality=85, every=1, queue_size=16, max_bytes=300_000_000):
        self.quality, self.every, self.max_bytes = quality, every, max_bytes
        self.encoded = []          # (frame_id, JPEG bytes)
        self.bytes = 0
        self.dropped_queue = 0     # Counted on the control thread.
        self.dropped_budget = 0    # Counted on the worker thread.
        self.queue = queue.Queue(queue_size)
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def add(self, frame_id, frame):
        """Never blocks. The caller must not modify `frame` afterwards."""
        if self.every < 1 or frame_id % self.every:
            return
        try:
            self.queue.put_nowait((frame_id, frame))
        except queue.Full:
            self.dropped_queue += 1

    def _work(self):
        import cv2
        while True:
            item = self.queue.get()
            if item is None:
                return
            frame_id, frame = item
            try:
                ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            except cv2.error:
                ok = False
            if not ok or self.bytes + len(data) > self.max_bytes:
                self.dropped_budget += 1
                continue
            self.bytes += len(data)
            self.encoded.append((frame_id, data.tobytes()))

    def close(self, timeout=10.0):
        """Finish encoding queued frames. Call only after motors are stopped."""
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=timeout)
            except queue.Full:
                return
            self.thread.join(timeout)

    def save(self, folder):
        self.close()
        folder.mkdir(exist_ok=True)
        for frame_id, data in list(self.encoded):
            (folder / f"frame_{frame_id:05d}.jpg").write_bytes(data)
        return {"recorded_frames": len(self.encoded), "dropped_frames":
                self.dropped_queue + self.dropped_budget, "recorded_mb": round(self.bytes / 1e6, 1),
                "jpeg_quality": self.quality, "record_every": self.every}
