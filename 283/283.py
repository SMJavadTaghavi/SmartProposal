from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class CrossRefCheckResult:
    ok: bool
    endpoint: str
    http_status: Optional[int]
    latency_ms: Optional[int]
    timed_out: bool
    attempts: int
    error: Optional[str]
    sample_payload_ok: bool


class CrossRefClient:
    """
    A small CrossRef API client focused on:
    - Connectivity verification
    - Timeout management (connect/read)
    - Retry policy for transient failures
    """

    def __init__(
        self,
        base_url: str = "https://api.crossref.org",
        user_agent: str = "SmartProposal/1.0 (mailto:unknown@example.com)",
        connect_timeout: float = 3.0,
        read_timeout: float = 6.0,
        total_retries: int = 3,
        backoff_factor: float = 0.6,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = (connect_timeout, read_timeout)

        # Create a requests Session for connection pooling + consistent headers.
        self.session = requests.Session()
        self.session.headers.update(
            {
                # CrossRef recommends setting a descriptive UA (and mailto if possible).
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

        # Configure retries for transient errors.
        # Important: timeouts raise exceptions; we handle them and optionally retry manually.
        # Here we retry on common transient HTTP statuses and network errors (via urllib3).
        retry = Retry(
            total=total_retries,
            connect=total_retries,
            read=total_retries,
            status=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _safe_get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Perform GET and attempt to parse JSON safely.
        Returns (json_dict_or_none, http_status_or_none).
        """
        resp = self.session.get(url, params=params, timeout=self.timeout)
        status = resp.status_code
        try:
            return resp.json(), status
        except Exception:
            return None, status

    def health_check(
        self,
        max_attempts: int = 3,
        jitter_ms: int = 250,
    ) -> CrossRefCheckResult:
        """
        Connectivity check strategy:
        1) Call a lightweight endpoint with a small result size: /works?rows=0
        2) Validate:
           - HTTP 200
           - JSON structure contains message/status or message fields
        3) Handle timeouts cleanly and retry a limited number of times.
        """
        endpoint = f"{self.base_url}/works"
        params = {"rows": 0}

        attempts = 0
        timed_out = False
        last_error: Optional[str] = None
        http_status: Optional[int] = None
        latency_ms: Optional[int] = None
        sample_payload_ok = False

        for i in range(max_attempts):
            attempts += 1
            start = time.time()

            try:
                payload, http_status = self._safe_get_json(endpoint, params=params)
                latency_ms = int((time.time() - start) * 1000)

                # Consider 200 a success; 429/5xx may indicate connectivity but rate-limit/server issues.
                if http_status == 200 and isinstance(payload, dict):
                    # CrossRef typically returns: {"status":"ok","message-type":"work-list","message":{...}}
                    # We'll check for minimal expected keys.
                    status_val = payload.get("status")
                    msg = payload.get("message")
                    sample_payload_ok = (status_val in ("ok", "available", None)) and isinstance(msg, dict)

                    if sample_payload_ok:
                        return CrossRefCheckResult(
                            ok=True,
                            endpoint=endpoint,
                            http_status=http_status,
                            latency_ms=latency_ms,
                            timed_out=False,
                            attempts=attempts,
                            error=None,
                            sample_payload_ok=True,
                        )

                # Not OK: capture a short error description
                last_error = f"Unexpected response: http_status={http_status}, payload_ok={bool(payload)}"
            except requests.exceptions.Timeout as e:
                latency_ms = int((time.time() - start) * 1000)
                timed_out = True
                last_error = f"Timeout: {type(e).__name__}"
            except requests.exceptions.RequestException as e:
                latency_ms = int((time.time() - start) * 1000)
                last_error = f"RequestException: {type(e).__name__}: {e}"
            except Exception as e:
                latency_ms = int((time.time() - start) * 1000)
                last_error = f"UnexpectedException: {type(e).__name__}: {e}"

            # Small jittered backoff between attempts to avoid thundering herd.
            # Even if adapter retries internally, this is a second safety layer.
            if i < max_attempts - 1:
                sleep_s = (random.randint(0, jitter_ms) / 1000.0) + (0.35 * (2 ** i))
                time.sleep(sleep_s)

        return CrossRefCheckResult(
            ok=False,
            endpoint=endpoint,
            http_status=http_status,
            latency_ms=latency_ms,
            timed_out=timed_out,
            attempts=attempts,
            error=last_error,
            sample_payload_ok=sample_payload_ok,
        )


def check_crossref(
    user_agent: str = "SmartProposal/1.0 (mailto:unknown@example.com)",
    connect_timeout: float = 3.0,
    read_timeout: float = 6.0,
    retries: int = 3,
    max_attempts: int = 3,
) -> CrossRefCheckResult:
    """
    Convenience function for callers.
    """
    client = CrossRefClient(
        user_agent=user_agent,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        total_retries=retries,
    )
    return client.health_check(max_attempts=max_attempts)


if __name__ == "__main__":
    # CLI usage:
    #   python crossref_healthcheck.py
    # Optional environment-specific flags can be added later; keeping it minimal for the task.
    result = check_crossref()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    # Exit code: 0 if ok, 2 otherwise (simple convention).
    raise SystemExit(0 if result.ok else 2)
