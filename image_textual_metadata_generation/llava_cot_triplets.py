"""
LLaVA-based Chain-of-Thought triplet generation for crisis images.

Generates a structured four-step CoT rationale (entity identification →
contextual grounding → category linking → decision) and parses it into a
per-instance crisis knowledge graph payload.

This module is completely independent of llava_captions.py and
llava_concept_triplets.py.  It has its own session, constants, service
class, and cache key ("cot_triplets").

Required: a running Ollama instance with the llava model.
  ollama serve && ollama pull llava
"""

import base64
import json
import os
import re
import signal
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Shared HTTP session (independent of other llava modules)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 90          # CoT output is longer than simple captions/triplets
MAX_RETRIES = 3            # More retries: JSON parsing may fail on first attempt
OLLAMA_KEEP_ALIVE = "30m"
COT_TRIPLET_MAX_TOKENS = 1024

# Structured prompt with a concrete example so LLaVA can mimic the format.
_COT_PROMPT = (
    "You are a crisis analysis expert. "
    "Given the image, reason step by step about the depicted crisis event.\n\n"
    "You MUST return ONLY a single JSON object (no arrays, no markdown, no extra text).\n"
    "Do NOT wrap the JSON in ```json or any code fences.\n\n"
    "Here is an example of the EXACT format you must follow:\n"
    '{"steps":["Step 1: The building shows severe structural collapse with walls crumbled.'
    '","Step 2: Debris and dust clouds are visible, along with displaced residents nearby.'
    '","Step 3: Structural collapse combined with geographic context suggests seismic activity.'
    '","Step 4: Most likely crisis type is earthquake with high confidence."],'
    '"entities":["collapsed building","debris","displaced residents"],'
    '"relations":[["building","damaged_by","earthquake"],["residents","displaced_from","building"]],'
    '"crisis_type":"earthquake"}\n\n'
    "Now analyze the provided image. Return ONLY a JSON object with these keys:\n"
    '- "steps": array of exactly 4 strings (your reasoning for Steps 1-4)\n'
    '- "entities": array of entity strings found in the image\n'
    '- "relations": array of [subject, predicate, object] triples\n'
    '- "crisis_type": one of earthquake, flood, landslide, fire, hurricane, other_disaster, not_disaster\n'
)

# Fallback template used when all JSON parse attempts fail.
_FALLBACK_STEP_TEMPLATES = [
    "Primary entity and physical state could not be determined from image.",
    "No contextual evidence extracted.",
    "Entity-to-category link unavailable.",
    "Crisis type undetermined; low confidence.",
]

VALID_CRISIS_TYPES = {
    "earthquake", "flood", "landslide", "fire",
    "hurricane", "other_disaster", "not_disaster",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_close(response) -> None:
    try:
        response.close()
    except Exception:
        pass


def _normalise_text(text: str) -> str:
    """Unicode quote normalisation + whitespace collapse."""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"```[a-z]*\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _is_degenerate_output(text: str) -> bool:
    """Detect degenerate repetitive LLM output (e.g. 'ThreadThreadThread...').

    Returns True if the output is clearly garbage and should be aborted.
    """
    if "{" not in text:
        return True

    stripped = _strip_markdown_fences(text)
    content_before_brace = stripped.split("{")[0]
    if len(content_before_brace) > 80:
        word_counts: dict = {}
        for word in re.findall(r"[A-Za-z]{3,}", content_before_brace):
            word_counts[word] = word_counts.get(word, 0) + 1
        if any(count >= 8 for count in word_counts.values()):
            return True

    if re.search(r"(Thread|thread){5,}", text):
        return True

    return False


def _extract_json_block(text: str) -> Optional[str]:
    """Try to isolate the first balanced {...} block from raw LLM output.

    Uses brace-depth counting so we don't greedily match across multiple
    unrelated JSON objects.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Truncated JSON -- return what we have and let the repair logic handle it
    return text[start:]


def _try_repair_truncated_json(text: str) -> Optional[str]:
    """Attempt to close unclosed brackets/braces in truncated JSON."""
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False
    last_valid = 0

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1
        last_valid = i

    if open_braces == 0 and open_brackets == 0:
        return None

    # Trim any trailing incomplete string value
    repaired = text[: last_valid + 1].rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    # Close any dangling string literal
    quote_count = repaired.count('"') - repaired.count('\\"')
    if quote_count % 2 == 1:
        repaired += '"'

    repaired += "]" * max(open_brackets, 0)
    repaired += "}" * max(open_braces, 0)
    return repaired


def _try_json_loads(text: str) -> Optional[Any]:
    """Try json.loads with progressive repairs for common LLM mistakes."""
    # 1) Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Strip trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3) Try repairing truncated JSON
    repaired = _try_repair_truncated_json(cleaned)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return None


def _unwrap_to_dict(data: Any) -> Optional[Dict]:
    """If `data` is a list containing a dict, unwrap it. Otherwise return a dict or None."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    return None


def _parse_cot_json(raw: str) -> Optional[Dict]:
    """
    Parse the LLM output into a structured dict.

    Returns a dict with keys: steps (list[str] of length 4), entities
    (list[str]), relations (list[list[str]]), crisis_type (str).
    Returns None if parsing fails.
    """
    text = _normalise_text(raw)
    text = _strip_markdown_fences(text)

    # Strategy 1: extract the first balanced {} block
    json_block = _extract_json_block(text)
    data = None
    if json_block is not None:
        parsed = _try_json_loads(json_block)
        data = _unwrap_to_dict(parsed)

    # Strategy 2: if that failed, maybe the whole output is a JSON array wrapping a dict
    if data is None:
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match:
            parsed = _try_json_loads(arr_match.group(0))
            data = _unwrap_to_dict(parsed)

    # Strategy 3: extract individual fields with regex as a last resort
    if data is None:
        data = _regex_extract_fields(text)

    if data is None:
        return None

    # --- Normalise the parsed dict into our canonical schema ---

    # Steps: accept "steps" key or "step"-keyed objects in an array
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        steps = [str(steps)]
    normalised_steps: List[str] = []
    for s in steps:
        if isinstance(s, dict):
            normalised_steps.append(
                str(s.get("step") or s.get("text") or s.get("context")
                    or s.get("linkage") or s.get("confidence") or next(iter(s.values()), ""))
            )
        else:
            normalised_steps.append(str(s).strip())
    while len(normalised_steps) < 4:
        normalised_steps.append(_FALLBACK_STEP_TEMPLATES[len(normalised_steps)])
    steps = normalised_steps[:4]

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    entities = [str(e).strip() for e in entities if str(e).strip()]

    relations = data.get("relations", [])
    if not isinstance(relations, list):
        relations = []
    valid_relations: List[List[str]] = []
    for r in relations:
        if isinstance(r, list) and len(r) == 3:
            s, p, o = str(r[0]).strip(), str(r[1]).strip(), str(r[2]).strip()
            if s and p and o:
                valid_relations.append([s, p, o])

    crisis_type = str(data.get("crisis_type", "")).strip().lower()
    if crisis_type not in VALID_CRISIS_TYPES:
        matched = next((ct for ct in VALID_CRISIS_TYPES if ct in crisis_type), "other_disaster")
        crisis_type = matched

    return {
        "steps": steps,
        "entities": entities,
        "relations": valid_relations,
        "crisis_type": crisis_type,
    }


def _regex_extract_fields(text: str) -> Optional[Dict]:
    """Last-resort field extraction using regex when JSON parsing totally fails."""
    result: Dict[str, Any] = {}

    steps_match = re.search(r'"steps"\s*:\s*\[([^\]]*)', text)
    if steps_match:
        raw_steps = re.findall(r'"([^"]+)"', steps_match.group(1))
        if raw_steps:
            result["steps"] = raw_steps

    entities_match = re.search(r'"entities"\s*:\s*\[([^\]]*)', text)
    if entities_match:
        result["entities"] = re.findall(r'"([^"]+)"', entities_match.group(1))

    crisis_match = re.search(r'"crisis_type"\s*:\s*"([^"]+)"', text)
    if crisis_match:
        result["crisis_type"] = crisis_match.group(1)

    relations_match = re.search(r'"relations"\s*:\s*\[(\[[\s\S]*?\])\s*\]', text)
    if relations_match:
        try:
            result["relations"] = json.loads("[" + relations_match.group(1) + "]")
        except json.JSONDecodeError:
            result["relations"] = []

    if not result:
        return None
    return result


def _fallback_from_text(raw: str) -> Dict:
    """
    Template-based extraction used when all JSON parse attempts fail.
    Scans raw text for crisis-type keywords and assembles minimal output.
    """
    text_lower = raw.lower()
    crisis_type = "other_disaster"
    for ct in VALID_CRISIS_TYPES:
        if ct.replace("_", " ") in text_lower or ct in text_lower:
            crisis_type = ct
            break

    # Try to pull entity-like nouns from the text (very rough heuristic)
    entity_candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", raw)
    entities = list(dict.fromkeys(entity_candidates))[:10]

    return {
        "steps": list(_FALLBACK_STEP_TEMPLATES),
        "entities": entities,
        "relations": [],
        "crisis_type": crisis_type,
    }


# ---------------------------------------------------------------------------
# Main generation class
# ---------------------------------------------------------------------------

class LlavaCoTTripletGeneration:
    """
    Generate a structured 4-step Chain-of-Thought rationale from a crisis
    image using LLaVA via Ollama.

    The output dict contains:
      - steps:       list[str] of length 4 (one per CoT step)
      - entities:    list[str] of crisis-relevant entities
      - relations:   list[[str, str, str]] of (subject, predicate, object) triples
      - crisis_type: str — predicted crisis category

    This class is completely independent of LlavaCaptionGeneration and
    LlavaCaptionConceptTripletExtraction.
    """

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.model = "llava"
        self.api_url = "http://localhost:11434/api/generate"

    def encode_image_to_base64(self) -> str:
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if os.path.getsize(self.image_path) == 0:
            raise ValueError(f"Image file is empty (0 bytes): {self.image_path}")
        with open(self.image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "images": [self.encode_image_to_base64()],
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_predict": COT_TRIPLET_MAX_TOKENS,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.3,
                "repeat_last_n": 256,
                "penalize_newline": False,
            },
        }

    def _stream_raw_text(self, prompt: str, cancel_event: threading.Event) -> str:
        """Send prompt to Ollama, stream response, return raw text.

        Aborts early if the model enters a degenerate repetition loop.
        """
        payload = self._build_payload(prompt)
        response = _SESSION.post(
            self.api_url,
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        raw = ""
        try:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if cancel_event and cancel_event.is_set():
                    return raw
                if not line:
                    continue
                try:
                    json_line = json.loads(line)
                    raw += json_line.get("response", "")
                    if json_line.get("done"):
                        break
                except json.JSONDecodeError:
                    raw += line

                if len(raw) > 100 and _is_degenerate_output(raw):
                    print("[cot_triplets] Aborting: degenerate/repetitive output detected.")
                    cancel_event.set()
                    return raw
        finally:
            _safe_close(response)

        if cancel_event and cancel_event.is_set():
            return raw

        print("[cot_triplets] LLM raw output:", raw[:500])
        return raw

    def get_cot_triplets(self, timeout_seconds: float = 0) -> Optional[Dict]:
        """
        Generate a CoT rationale dict with up to MAX_RETRIES attempts.
        Returns None if generation fails so that the result is not cached
        and the sample can be retried later.

        Uses a cooperative cancel_event for streaming timeout control,
        plus a SIGALRM hard backstop on the main thread.
        """
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        cancel_event = threading.Event()
        is_main = threading.current_thread() is threading.main_thread()

        for attempt in range(MAX_RETRIES):
            if deadline and time.monotonic() >= deadline:
                return None

            cancel_event.clear()
            remaining = (deadline - time.monotonic()) if deadline else READ_TIMEOUT + 30
            if remaining <= 0:
                return None

            prev_handler = None
            if is_main:
                def _alarm_handler(signum, frame):
                    cancel_event.set()
                    raise TimeoutError("SIGALRM hard timeout (cot_triplets)")
                prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(int(remaining) + 5)

            timer = threading.Timer(remaining, cancel_event.set)
            timer.daemon = True
            timer.start()

            try:
                if cancel_event.is_set():
                    break
                raw = self._stream_raw_text(_COT_PROMPT, cancel_event)
                if cancel_event.is_set():
                    print(f"[cot_triplets] Cancelled during streaming on attempt {attempt + 1}/{MAX_RETRIES}")
                    continue
                parsed = _parse_cot_json(raw)
                if parsed is not None:
                    return parsed
                print(
                    f"[cot_triplets] JSON parse failed on attempt {attempt + 1}/{MAX_RETRIES}, re-prompting..."
                )
            except TimeoutError:
                print(f"[cot_triplets] HARD TIMEOUT attempt {attempt + 1}/{MAX_RETRIES}")
            except (requests.ConnectionError, requests.Timeout) as exc:
                print(f"[cot_triplets] NETWORK ERROR attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            except Exception as exc:
                print(f"[cot_triplets] ERROR attempt {attempt + 1}/{MAX_RETRIES}: {exc}")
            finally:
                timer.cancel()
                cancel_event.set()
                if is_main and prev_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev_handler)

            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)

        print("[cot_triplets] All JSON parse attempts failed; returning None (will not cache).")
        return None
