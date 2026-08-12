"""
Retry, timeout and connection-reuse tests for ApiClient.

These run against a stdlib HTTP server on loopback rather than requests_mock,
because requests_mock replaces the transport adapter and therefore never
exercises the HTTPAdapter/urllib3 Retry layer that is under test here.
No traffic leaves the machine.
"""

import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from codecarbon.core.api_client import ApiClient

CONF = {
    "os": "linux",
    "python_version": "3.12",
    "codecarbon_version": "3.0",
    "cpu_count": 8,
    "cpu_model": "CPU",
    "gpu_count": 0,
    "gpu_model": "",
    "longitude": 0.0,
    "latitude": 0.0,
    "region": "EU",
    "provider": "none",
    "ram_total_size": 16.0,
    "tracking_mode": "machine",
}

EMISSION = {
    "duration": 5,
    "emissions": 1.0,
    "emissions_rate": 1.0,
    "cpu_power": 1.0,
    "gpu_power": 0.0,
    "ram_power": 0.5,
    "cpu_energy": 0.1,
    "gpu_energy": 0.0,
    "ram_energy": 0.1,
    "energy_consumed": 0.2,
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so pooling is observable

    def log_message(self, *args):
        pass

    def _serve(self):
        state = self.server.state
        state["requests"] += 1
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        if state["hang"]:
            # Answer far too late, i.e. the client hits its read timeout while
            # the server is happily processing the request.
            time.sleep(state["hang"])
        plan = state["status_plan"]
        code = plan.pop(0) if plan else state["success_status"]
        body = b'{"id": "run-1"}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _serve
    do_POST = _serve


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state):
        self.state = state
        super().__init__(("127.0.0.1", 0), _Handler)

    def process_request(self, request, client_address):
        self.state["connections"] += 1
        super().process_request(request, client_address)


class StubServerTestCase(unittest.TestCase):
    """A loopback HTTP server whose responses can be scripted per test."""

    def setUp(self):
        self.state = {
            "requests": 0,
            "connections": 0,
            "status_plan": [],
            "success_status": 201,
            "hang": 0,
        }
        self.server = _Server(self.state)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def client(self, **kwargs):
        kwargs.setdefault("retries", 2)
        kwargs.setdefault("backoff", 0)  # keep the suite fast
        api = ApiClient(
            endpoint_url=self.url,
            experiment_id="exp-1",
            conf=CONF,
            create_run_automatically=False,
            **kwargs,
        )
        self.addCleanup(api.close)
        api.run_id = "run-1"
        return api


class TestRetry(StubServerTestCase):
    def test_retries_then_succeeds_on_transient_503(self):
        self.state["status_plan"] = [503, 503]
        api = self.client()

        self.assertTrue(api.add_emission(dict(EMISSION)))
        self.assertEqual(self.state["requests"], 3)

    def test_gives_up_after_configured_retries(self):
        self.state["status_plan"] = [503] * 10
        api = self.client(retries=2)

        with self.assertRaises(requests.exceptions.HTTPError):
            api.add_emission(dict(EMISSION))
        self.assertEqual(self.state["requests"], 3)  # 1 attempt + 2 retries

    def test_no_retry_on_client_error(self):
        """Retrying a validation error is pure waste."""
        self.state["status_plan"] = [400] * 10
        api = self.client()

        with self.assertRaises(requests.exceptions.HTTPError):
            api.add_emission(dict(EMISSION))
        self.assertEqual(self.state["requests"], 1)

    def test_retries_on_connection_error(self):
        """Nothing is listening, so every attempt fails to connect."""
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        api = ApiClient(
            endpoint_url=f"http://127.0.0.1:{port}",
            experiment_id="exp-1",
            conf=CONF,
            create_run_automatically=False,
            retries=2,
            backoff=0,
        )
        self.addCleanup(api.close)
        api.run_id = "run-1"

        with self.assertRaises(requests.exceptions.ConnectionError) as ctx:
            api.add_emission(dict(EMISSION))
        # urllib3 reports the exhausted budget in the message.
        self.assertIn("max retries exceeded", str(ctx.exception).lower())


class TestPostIsNotReplayedAfterTheRequestLanded(StubServerTestCase):
    """
    carbonserver has no idempotency key and the dashboard sums emission rows,
    so replaying a POST whose response was lost silently inflates a user's
    reported emissions. Only failures that plausibly never reached the
    application may be retried on POST.
    """

    def test_read_timeout_on_post_is_not_retried(self):
        self.state["hang"] = 3  # much longer than the read timeout below
        api = self.client(timeout=(3.05, 0.3))

        with self.assertRaises(requests.exceptions.ReadTimeout):
            api.add_emission(dict(EMISSION))
        self.assertEqual(self.state["requests"], 1)

    def test_503_on_post_is_still_retried(self):
        """The counterpart: 503 means no upstream took the request."""
        self.state["status_plan"] = [503, 503]
        api = self.client(timeout=(3.05, 0.3))

        self.assertTrue(api.add_emission(dict(EMISSION)))
        self.assertEqual(self.state["requests"], 3)

    def test_504_on_post_is_not_retried(self):
        """A gateway timeout means the app was reached and may have committed."""
        self.state["status_plan"] = [504] * 10
        api = self.client()

        with self.assertRaises(requests.exceptions.HTTPError):
            api.add_emission(dict(EMISSION))
        self.assertEqual(self.state["requests"], 1)

    def test_read_timeout_on_get_is_still_retried(self):
        """GETs are idempotent, so they keep the broader policy."""
        self.state["hang"] = 3
        api = self.client(timeout=(3.05, 0.3))

        with self.assertRaises(requests.exceptions.RequestException):
            api.get_list_organizations()
        self.assertEqual(self.state["requests"], 3)


class TestSessionReuse(StubServerTestCase):
    def test_sequential_calls_reuse_one_connection(self):
        api = self.client()

        for _ in range(5):
            self.assertTrue(api.add_emission(dict(EMISSION)))

        self.assertEqual(self.state["requests"], 5)
        self.assertEqual(self.state["connections"], 1)

    def test_close_releases_the_session(self):
        api = self.client()
        api.add_emission(dict(EMISSION))
        api.close()
        api.close()  # idempotent


class TestTimeoutConfiguration(unittest.TestCase):
    def test_timeout_is_passed_through_to_requests(self):
        api = ApiClient(
            endpoint_url="http://test.com",
            create_run_automatically=False,
            timeout=(1.5, 7),
        )
        seen = {}

        def fake_get(url, json, timeout, headers):
            seen["timeout"] = timeout
            return type("R", (), {"status_code": 200, "json": lambda self: {}})()

        api._request(fake_get, "http://test.com/x")
        self.assertEqual(seen["timeout"], (1.5, 7))

    def test_default_timeout_is_not_the_old_hardcoded_two_seconds(self):
        api = ApiClient(endpoint_url="http://test.com", create_run_automatically=False)
        self.assertEqual(api._timeout, (3.05, 10))


class TestCreateRunBackoff(StubServerTestCase):
    def test_failed_run_creation_is_not_retried_on_every_tick(self):
        self.state["status_plan"] = [400] * 10  # not retryable, fails fast
        api = ApiClient(
            endpoint_url=self.url,
            experiment_id="exp-1",
            conf=CONF,
            create_run_automatically=False,
            retries=0,
        )
        self.addCleanup(api.close)

        with self.assertRaises(requests.exceptions.HTTPError):
            api._create_run("exp-1")
        self.assertEqual(self.state["requests"], 1)

        # Subsequent attempts are suppressed while the backoff is in effect.
        for _ in range(5):
            self.assertIsNone(api._create_run("exp-1"))
        self.assertEqual(self.state["requests"], 1)

    def test_backoff_clears_after_a_success(self):
        api = ApiClient(
            endpoint_url=self.url,
            experiment_id="exp-1",
            conf=CONF,
            create_run_automatically=False,
            retries=0,
        )
        self.addCleanup(api.close)

        self.assertEqual(api._create_run("exp-1"), "run-1")
        self.assertEqual(api._create_run_not_before, 0.0)


if __name__ == "__main__":
    unittest.main()
