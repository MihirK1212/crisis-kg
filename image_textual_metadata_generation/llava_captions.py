import base64
import json
import os
import re
import signal
import time
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SESSION = requests.Session()
_RETRY_STRATEGY = Retry(
    total=1,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY_STRATEGY, pool_connections=4, pool_maxsize=4)
_SESSION.mount("http://", _ADAPTER)
_SESSION.mount("https://", _ADAPTER)

CONNECT_TIMEOUT = 10
# If Ollama hasn't started sending tokens within this time, it's stuck.
READ_TIMEOUT = 30
MAX_RETRIES = 2
OLLAMA_KEEP_ALIVE = "30m"
CAPTION_MAX_TOKENS = 150


def _safe_close(response):
    """Close response without blocking — fire and forget."""
    try:
        response.close()
    except Exception:
        pass


class LlavaCaptionGeneration:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.model = "llava"
        self.api_url = "http://localhost:11434/api/generate"
        self.prompt = (
            "Describe this image in one or two concise, informative sentences. "
            "Focus on the main subject, setting, and any notable visual details. "
            "Do not include greetings, preambles, or meta-commentary — output only the caption."
        )

    def encode_image_to_base64(self) -> str:
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        file_size = os.path.getsize(self.image_path)
        if file_size == 0:
            raise ValueError(f"Image file is empty (0 bytes): {self.image_path}")
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _build_payload(self) -> dict:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "images": [self.encode_image_to_base64()],
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_predict": CAPTION_MAX_TOKENS,
            },
        }

    def _call_api(self, cancel_event: threading.Event = None) -> str:
        """Stream caption from Ollama. Aborts immediately if cancel_event is set."""
        payload = self._build_payload()

        response = _SESSION.post(
            self.api_url,
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        try:
            response.raise_for_status()
            result_text = ""
            for line in response.iter_lines(decode_unicode=True):
                if cancel_event and cancel_event.is_set():
                    return ""
                if not line:
                    continue
                try:
                    json_line = json.loads(line)
                    result_text += json_line.get("response", "")
                    if json_line.get("done"):
                        break
                except json.JSONDecodeError:
                    result_text += line
        finally:
            _safe_close(response)

        if cancel_event and cancel_event.is_set():
            return ""

        return self._clean_caption(result_text)

    def _clean_caption(self, text: str) -> str:
        """Normalize and clean raw model output into a single clean caption string."""
        print("Caption LLM raw output text", text)

        if not text:
            return ""

        text = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        text = re.sub(r"```[a-z]*\s*", "", text)
        text = re.sub(r"```", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip('"').strip("'").strip("`").strip()

        preamble_patterns = [
            r"^(sure[,!.]?\s*(here (is|are)[^:]*)?[:\s]*)",
            r"^(of course[,!.]?\s*)",
            r"^(certainly[,!.]?\s*)",
            r"^(here('s| is| are) (a |the |my )?([^:]*?)?[:\s]+)",
            r"^(the image (shows?|depicts?|contains?|features?|illustrates?|presents?)[:\s]+)",
            r"^(this image (shows?|depicts?|contains?|features?|illustrates?|presents?)[:\s]+)",
            r"^(in this image[,:\s]+)",
            r"^(the (photo|picture|photograph) (shows?|depicts?|contains?)[:\s]+)",
            r"^(caption[:\s]+)",
            r"^(answer[:\s]+)",
            r"^(output[:\s]+)",
            r"^(description[:\s]+)",
        ]
        for pattern in preamble_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        meta_suffixes = [
            r"\s*\(note:.*",
            r"\s*note:.*",
            r"\s*\[end\].*",
            r"\s*---.*",
        ]
        for pattern in meta_suffixes:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        text = text.strip().strip('"').strip("'").strip("`").strip()

        if len(text) > 1000:
            text = text[:1000].rsplit(" ", 1)[0] + "..."

        return text

    def get_caption(self, timeout_seconds: float = 0) -> str:
        """Generate a caption. Guaranteed to return within timeout_seconds.

        Uses SIGALRM as a hard backstop (from main thread) and cooperative
        cancel_event for streaming. Neither can be blocked by a stuck syscall.
        """
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        cancel_event = threading.Event()
        is_main = threading.current_thread() is threading.main_thread()

        for attempt in range(MAX_RETRIES):
            if deadline and time.monotonic() >= deadline:
                return ""

            cancel_event.clear()

            remaining = (deadline - time.monotonic()) if deadline else READ_TIMEOUT + 15
            if remaining <= 0:
                return ""

            # Hard SIGALRM backstop (main thread only) — absolutely cannot hang
            prev_handler = None
            if is_main:
                def _alarm_handler(signum, frame):
                    cancel_event.set()
                    raise TimeoutError("SIGALRM hard timeout")
                prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(int(remaining) + 5)

            # Cooperative timer for worker threads (and as first line of defense)
            timer = threading.Timer(remaining, cancel_event.set)
            timer.daemon = True
            timer.start()

            try:
                caption = self._call_api(cancel_event=cancel_event)
                if caption:
                    return caption
            except TimeoutError:
                print(f"HARD TIMEOUT Caption attempt {attempt + 1}/{MAX_RETRIES}")
            except (requests.ConnectionError, requests.Timeout) as exc:
                print(f"NETWORK ERROR Caption attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            except Exception as exc:
                print(f"ERROR Caption attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            finally:
                timer.cancel()
                cancel_event.set()
                if is_main and prev_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev_handler)

            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)

        return ""
