"""One running inference and at most one pending newest frame."""
import queue
import threading
import time


def replace_latest(channel, item):
    try:
        channel.get_nowait()
    except queue.Empty:
        pass
    channel.put_nowait(item)


class LatestWorker:
    def __init__(self, reader):
        self.reader = reader
        self.inputs, self.results = queue.Queue(1), queue.Queue(1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, generation, stamp, frame):
        replace_latest(self.inputs, (generation, stamp, frame))

    def run(self):
        while not self.stop_event.is_set():
            try:
                generation, stamp, frame = self.inputs.get(timeout=.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            value, error = None, None
            try:
                value = self.reader.read(frame)
            except Exception as exc:
                error = str(exc)
            result = (generation, stamp, frame, value, error, time.monotonic() - started)
            if not self.stop_event.is_set():
                replace_latest(self.results, result)

    def stop(self):
        self.stop_event.set()
