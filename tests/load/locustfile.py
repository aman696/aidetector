"""
Load test for the detector API — "how many users at once can my server handle?"

One command (from the repo root, with the app running):

    .venv/bin/locust -f tests/load/locustfile.py --headless \
        -u 20 -r 2 -t 2m --host http://localhost:8000

    -u   peak number of concurrent users to ramp to
    -r   users spawned per second (ramp rate)
    -t   test duration (e.g. 2m, 30s)
    --host   the server under test

Env knobs (optional):
    LOCUST_IMAGE     path to a real image to upload (default: a generated 768x768 JPEG)
    LOCUST_MIN_WAIT  / LOCUST_MAX_WAIT   per-user think time in seconds
                     (default 0/0 = closed-loop, hammer as fast as replies come back)

IMPORTANT — for a TRUE capacity test, relax the app's own guard rails first, or you
will measure the rate limiter and the concurrency cap, not the hardware. Run the
server with:

    RATE_LIMIT=100000/minute MAX_CONCURRENT=8 MAX_QUEUE=200 \
        .venv/bin/python app.py

Then ramp -u until p95 latency crosses what you can tolerate or failures climb — that
user count (and the achieved requests/sec) is your answer.
"""

import io
import os

from locust import HttpUser, task, between


def _payload() -> bytes:
    """Image bytes to upload: a real file if LOCUST_IMAGE is set, else a generated
    768x768 JPEG so the test needs no dataset on disk."""
    path = os.getenv("LOCUST_IMAGE", "")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    from PIL import Image
    import numpy as np
    arr = (np.random.default_rng(0).random((768, 768, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


_IMAGE_BYTES = _payload()
_MIN_WAIT = float(os.getenv("LOCUST_MIN_WAIT", "0"))
_MAX_WAIT = float(os.getenv("LOCUST_MAX_WAIT", "0"))


class DetectUser(HttpUser):
    wait_time = between(_MIN_WAIT, _MAX_WAIT)

    @task
    def detect(self):
        files = {"file": ("load.jpg", io.BytesIO(_IMAGE_BYTES), "image/jpeg")}
        with self.client.post("/api/detect", files=files, catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            elif r.status_code in (429, 503):
                # Backpressure (rate limit / queue full / timeout) — the server
                # shedding load on purpose, not a crash. Recorded as a failure so
                # it shows up, but labelled so you can tell it apart from a 500.
                r.failure(f"backpressure {r.status_code}")
            else:
                r.failure(f"HTTP {r.status_code}")
