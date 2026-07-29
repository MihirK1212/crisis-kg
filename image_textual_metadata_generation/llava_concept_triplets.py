import base64
import json
import os
import re
import signal
import time
import threading
from typing import List, Tuple

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
READ_TIMEOUT = 30
MAX_RETRIES = 2
OLLAMA_KEEP_ALIVE = "30m"
TRIPLET_MAX_TOKENS = 300


def _safe_close(response):
    """Close response without blocking — fire and forget."""
    try:
        response.close()
    except Exception:
        pass


class LlavaCaptionConceptTripletExtraction:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.model = "llava"
        self.api_url = "http://localhost:11434/api/generate"

        self.prompt = (
            "For the given image, output ONLY a single-line list of (subject,predicate,object) triplets "
            "describing concrete, visible facts in the image. Format: (s,p,o),(s,p,o),... "
            "No extra text, quotes, or line breaks. Subjects/objects are singular visible nouns; "
            "no pronouns/articles. Predicates are simple relations/attributes (on, under, next to, "
            "holding, wearing, color, material, number). Return up to 12 of the most salient triplets. avoid speculation."
            "Formatting constraints (must follow to ensure parsing):"
            "- No quotes, no brackets other than parentheses, no trailing periods"
            "- No explanatory text, headers, or labels"
            "- No line breaks; single-line output only"
            "Good examples:"
            "Image: [A man riding a horse in a field]"
            "Output: (man,riding,horse),(man,in,field),(horse,in,field)"
            "Image: [A woman holding a red umbrella on a rainy street]"
            "Output: (woman,holding,umbrella),(umbrella,color,red),(woman,in,street),(street,condition,rainy)"
            "Image: [Two boys playing soccer; one boy wearing a blue shirt]"
            "Output: (boys,playing,soccer),(boy,wearing,blue shirt),(boys,in,field),(ball,on,grass)"
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
                "num_predict": TRIPLET_MAX_TOKENS,
            },
        }

    def _call_api(self, cancel_event: threading.Event = None) -> list:
        """Stream triplets from Ollama. Aborts immediately if cancel_event is set."""
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
                    return []
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
            return []

        return self._extract_triplets_from_text(result_text)

    _MAX_TOKEN_LENGTH = 80
    _MAX_TRIPLETS = 20

    def _extract_triplets_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Robustly extract (subject, predicate, object) triplets from text."""
        print("Caption Triplets LLM raw output text", text)

        if not text:
            return []

        normalized = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        normalized = re.sub(r"```[a-z]*\s*", "", normalized)
        normalized = re.sub(r"```", "", normalized)
        normalized = re.sub(r"(?:^|\n)\s*(?:\d+[\.\)]\s*|-\s*)", " ", normalized)
        normalized = re.sub(
            r"^.*?(?:output|answer|triplets?|result)\s*:\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()

        bracket_pattern = r"[\(\[{]([^\)\]}]+)[\)\]}]"
        candidates = re.findall(bracket_pattern, normalized)

        triplets: List[Tuple[str, str, str]] = []

        for candidate in candidates:
            parts = [self._clean_token(p) for p in candidate.split(",")]

            if len(parts) < 3:
                continue
            if len(parts) > 3:
                first, second, *rest = parts
                third = ",".join(rest).strip()
                parts = [first, second, self._clean_token(third)]

            subject, predicate, obj = parts

            if not subject or not predicate or not obj:
                continue
            if any(
                len(tok) > self._MAX_TOKEN_LENGTH
                for tok in (subject, predicate, obj)
            ):
                continue
            if any(
                re.search(r"[;:\[\]{}]", tok)
                for tok in (subject, predicate, obj)
            ):
                continue
            if re.match(r"^\d+[\.\)]", subject):
                continue

            triplets.append((subject, predicate, obj))

            if len(triplets) >= self._MAX_TRIPLETS:
                break

        seen: set = set()
        unique_triplets: List[Tuple[str, str, str]] = []
        for t in triplets:
            key = (t[0].lower(), t[1].lower(), t[2].lower())
            if key in seen:
                continue
            seen.add(key)
            unique_triplets.append(t)

        return unique_triplets

    @staticmethod
    def _clean_token(token: str) -> str:
        token = token.strip().strip('"').strip("'").strip("`")
        token = token.replace("\n", " ")
        token = re.sub(r"\s+", " ", token)
        token = token.strip(".,;:!?")
        return token.strip()

    def get_triplets(self, timeout_seconds: float = 0) -> list:
        """Generate triplets. Guaranteed to return within timeout_seconds.

        Uses SIGALRM as a hard backstop (from main thread) and cooperative
        cancel_event for streaming.
        """
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        cancel_event = threading.Event()
        is_main = threading.current_thread() is threading.main_thread()

        for attempt in range(MAX_RETRIES):
            if deadline and time.monotonic() >= deadline:
                return []

            cancel_event.clear()

            remaining = (deadline - time.monotonic()) if deadline else READ_TIMEOUT + 15
            if remaining <= 0:
                return []

            prev_handler = None
            if is_main:
                def _alarm_handler(signum, frame):
                    cancel_event.set()
                    raise TimeoutError("SIGALRM hard timeout")
                prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(int(remaining) + 5)

            timer = threading.Timer(remaining, cancel_event.set)
            timer.daemon = True
            timer.start()

            try:
                triplets = self._call_api(cancel_event=cancel_event)
                if triplets:
                    return triplets
            except TimeoutError:
                print(f"HARD TIMEOUT Triplets attempt {attempt + 1}/{MAX_RETRIES}")
            except (requests.ConnectionError, requests.Timeout) as exc:
                print(f"NETWORK ERROR Triplets attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            except Exception as exc:
                print(f"ERROR Triplets attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            finally:
                timer.cancel()
                cancel_event.set()
                if is_main and prev_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev_handler)

            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)

        return []
