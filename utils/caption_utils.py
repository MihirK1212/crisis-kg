import json
import os
import tempfile
import threading
from typing import Any, Callable, Dict, Optional
from transformers import RobertaTokenizer

from image_textual_metadata_generation.llava_captions import LlavaCaptionGeneration
from image_textual_metadata_generation.llava_concept_triplets import (
    LlavaCaptionConceptTripletExtraction,
)
from image_textual_metadata_generation.llava_cot_triplets import LlavaCoTTripletGeneration
from utils import config_utils, data_utils, gpu_utils
import constants

config = config_utils.load_config()

# In-memory cache to avoid reloading JSON files repeatedly (keyed by file path)
_JSON_CACHE: Dict[str, Dict[str, Any]] = {}

# Single lock protects both the in-memory cache and the disk read-merge-write
# cycle so concurrent threads cannot race on either.
_JSON_LOCK = threading.Lock()

# # Logging control for generation/reuse messages
# _CAPTION_LOGGING_ENABLED = "caption_generation" in config.get(
#     constants.FIELD_MODEL_TO_USE
# )
_CAPTION_LOGGING_ENABLED = True

def set_caption_logging(enabled: bool) -> None:
    """Enable/disable caption logging prints at runtime."""
    global _CAPTION_LOGGING_ENABLED
    _CAPTION_LOGGING_ENABLED = bool(enabled)


def _load_json(json_file_path: str) -> Dict[str, Any]:
    """Thread-safe load of JSON from cache or disk (once per file path).

    Handles corrupted JSON gracefully: backs up the corrupted file and starts
    fresh so a single bad write doesn't permanently break the pipeline.
    """
    cached = _JSON_CACHE.get(json_file_path)
    if cached is not None:
        return cached

    with _JSON_LOCK:
        cached = _JSON_CACHE.get(json_file_path)
        if cached is not None:
            return cached

        data: Dict[str, Any] = {}
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                if not isinstance(data, dict):
                    print(
                        f"WARNING: JSON at '{json_file_path}' is not a dict "
                        f"(got {type(data).__name__}). Treating as empty."
                    )
                    data = {}
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                backup_path = json_file_path + ".corrupted"
                print(
                    f"WARNING: Corrupted JSON at '{json_file_path}': {exc}. "
                    f"Backing up to '{backup_path}' and starting fresh."
                )
                try:
                    os.replace(json_file_path, backup_path)
                except OSError as rename_err:
                    print(f"WARNING: Could not rename corrupted file: {rename_err}")
                data = {}
            except OSError as exc:
                print(f"WARNING: Could not read '{json_file_path}': {exc}. Starting fresh.")
                data = {}

        _JSON_CACHE[json_file_path] = data
        return data


def _ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)


def get_caption_from_image_file_path(image_file_path, timeout_seconds=0):
    return LlavaCaptionGeneration(image_file_path).get_caption(timeout_seconds=timeout_seconds)


def get_caption_concept_triplets_from_image_file_path(image_file_path, timeout_seconds=0):
    return LlavaCaptionConceptTripletExtraction(image_file_path).get_triplets(timeout_seconds=timeout_seconds)


def get_cot_triplets_from_image_file_path(image_file_path, timeout_seconds=0):
    return LlavaCoTTripletGeneration(image_file_path).get_cot_triplets(timeout_seconds=timeout_seconds)


class JsonBackedGenerationService:
    """
    Generic JSON-backed generator. Returns cached value for an image id if present,
    otherwise uses the provided generator to create, store, and return the value.
    """

    def __init__(
        self,
        *,
        json_dir: str,
        file_prefix: str,
        value_key: str,
        generator_fn: Callable,
        validator_fn: Optional[Callable] = None,
    ) -> None:
        self.json_dir = json_dir
        self.file_prefix = file_prefix
        self.value_key = value_key
        self.generator_fn = generator_fn
        self.validator_fn = validator_fn

    def _json_file_path(self) -> str:
        dataset_name = config.get(constants.FIELD_DATASET_TO_USE)
        base_name = f"{self.file_prefix}_{dataset_name}"
        if config[constants.FIELD_RUN_PARALLEL]:
            base_name = f"{base_name}_{int(gpu_utils.get_device())}"
        return os.path.join(self.json_dir, f"{base_name}.json")

    def get(
        self,
        *,
        image_id: str,
        image_file_path: str,
        update_forcefully: bool = False,
        log: Optional[bool] = None,
        timeout_seconds: float = 0,
    ) -> Any:
        log_enabled = _CAPTION_LOGGING_ENABLED if log is None else bool(log)

        json_file_path = self._json_file_path()

        # Fast-path: check in-memory cache without acquiring the write lock.
        # Also validates the stored value is non-empty so failed prior runs don't
        # permanently block re-generation.
        data = _load_json(json_file_path)
        if (
            (not update_forcefully)
            and (image_id in data)
            and (self.value_key in data[image_id])
            and data[image_id][self.value_key]
        ):
            cached_value = data[image_id][self.value_key]
            if self.validator_fn is None or self.validator_fn(cached_value):
                if log_enabled:
                    print(
                        f"[generation:{self.value_key}] Reusing existing for id='{image_id}'."
                    )
                return cached_value
            else:
                if log_enabled:
                    print(
                        f"[generation:{self.value_key}] Cached value for id='{image_id}' failed validation, regenerating."
                    )

        # Generate outside the lock — this is the slow LLaVA API call and can run
        # concurrently for different image IDs.
        if log_enabled:
            action = (
                "Regenerating"
                if image_id in data and update_forcefully
                else "Generating"
            )
            print(f"[generation:{self.value_key}] {action} for id='{image_id}'.")
        value = self.generator_fn(image_file_path, timeout_seconds=timeout_seconds)

        # Do not persist empty/failed results so the next run will retry them.
        if not value:
            print(
                f"[generation:{self.value_key}] Empty result for id='{image_id}', skipping cache."
            )
            return value

        # Acquire the lock for the read-merge-write cycle.  Re-reading from
        # disk inside the lock ensures we never overwrite data written by another
        # thread between our generation and our write.
        with _JSON_LOCK:
            fresh_data: Dict[str, Any] = {}
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        fresh_data = loaded
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError) as exc:
                    print(
                        f"WARNING: Could not read '{json_file_path}' during write-back: {exc}. "
                        f"Proceeding with in-memory cache."
                    )
                    fresh_data = dict(_JSON_CACHE.get(json_file_path, {}))

            fresh_data[image_id] = {
                self.value_key: value,
                "generation_time_stamp": data_utils.get_timestamp(),
            }

            _JSON_CACHE[json_file_path] = fresh_data

            # Atomic write: write to a temp file in the same directory, then
            # os.replace() to the target path.  This prevents half-written
            # JSON if the process is killed mid-write.
            _ensure_parent_dir(json_file_path)
            dir_name = os.path.dirname(json_file_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as tmp_file:
                    json.dump(fresh_data, tmp_file, indent=4)
                os.replace(tmp_path, json_file_path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        if log_enabled:
            print(
                f"[generation:{self.value_key}] Stored for id='{image_id}' at '{json_file_path}'."
            )
        return value

    def check_if_exists(self, image_id: str) -> bool:
        json_file_path = self._json_file_path()
        data = _load_json(json_file_path)
        if not (
            image_id in data
            and self.value_key in data[image_id]
            and bool(data[image_id][self.value_key])
        ):
            return False
        if self.validator_fn is not None:
            return self.validator_fn(data[image_id][self.value_key])
        return True


# Instantiate services for captions and concept triplets in separate folders/files
_CAPTION_SERVICE = JsonBackedGenerationService(
    json_dir="./image_textual_metadata_generation/captions",
    file_prefix="captions",
    value_key="caption",
    generator_fn=get_caption_from_image_file_path,
)

_TRIPLET_SERVICE = JsonBackedGenerationService(
    json_dir="./image_textual_metadata_generation/triplets",
    file_prefix="triplets",
    value_key="triplets",
    generator_fn=get_caption_concept_triplets_from_image_file_path,
)

def _validate_cot_triplet(value: Any) -> bool:
    """Reject cached CoT results that are clearly garbage/fallback.

    A valid CoT result must have at least one entity OR one relation.
    Results with empty entities AND empty relations were produced by the old
    fallback path and should be regenerated.
    """
    if not isinstance(value, dict):
        return False
    entities = value.get("entities", [])
    relations = value.get("relations", [])
    if not entities and not relations:
        return False
    return True


# CoT triplets service — independent cache directory and value key.
# Stores the full structured dict {steps, entities, relations, crisis_type}
# produced by LlavaCoTTripletGeneration.
_COT_TRIPLET_SERVICE = JsonBackedGenerationService(
    json_dir="./image_textual_metadata_generation/cot_triplets",
    file_prefix="cot_triplets",
    value_key="cot_triplets",
    generator_fn=get_cot_triplets_from_image_file_path,
    validator_fn=_validate_cot_triplet,
)


def get_caption(image_id, image_file_path, update_forcefully=False, log=None, timeout_seconds=0):
    if "caption_generation" in config.get(constants.FIELD_MODEL_TO_USE):
        assert (
            config.get(constants.FIELD_MODEL_TO_USE)
            == f"{config.get(constants.FIELD_DATASET_TO_USE)}_caption_generation"
        )
    return _CAPTION_SERVICE.get(
        image_id=str(image_id),
        image_file_path=image_file_path,
        update_forcefully=update_forcefully,
        log=log,
        timeout_seconds=timeout_seconds,
    )


def get_concept_triplets(image_id, image_file_path, update_forcefully=False, log=None, timeout_seconds=0):
    if "caption_generation" in config.get(constants.FIELD_MODEL_TO_USE):
        assert (
            config.get(constants.FIELD_MODEL_TO_USE)
            == f"{config.get(constants.FIELD_DATASET_TO_USE)}_caption_generation"
        )
    return _TRIPLET_SERVICE.get(
        image_id=str(image_id),
        image_file_path=image_file_path,
        update_forcefully=update_forcefully,
        log=log,
        timeout_seconds=timeout_seconds,
    )

def check_if_caption_triplet_exists(image_id: str) -> bool:
    return _TRIPLET_SERVICE.check_if_exists(image_id)

def check_if_caption_exists(image_id: str) -> bool:
    return _CAPTION_SERVICE.check_if_exists(image_id)


def get_cot_triplets(image_id, image_file_path, update_forcefully=False, log=None, timeout_seconds=0):
    """Return the cached (or freshly generated) CoT triplet dict for an image.

    The returned dict has keys: steps (list[str]), entities (list[str]),
    relations (list[[s,p,o]]), crisis_type (str).
    """
    return _COT_TRIPLET_SERVICE.get(
        image_id=str(image_id),
        image_file_path=image_file_path,
        update_forcefully=update_forcefully,
        log=log,
        timeout_seconds=timeout_seconds,
    )


def check_if_cot_triplet_exists(image_id: str) -> bool:
    return _COT_TRIPLET_SERVICE.check_if_exists(image_id)


def get_default_tokenizer():
    return RobertaTokenizer.from_pretrained(
        "roberta-base", truncation=True, do_lower_case=True
    )


from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download("stopwords")


def clean_text(text):
    import re
    import string

    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in ["rt", "st", "gt"]]

    return " ".join(words)


def pad_caption(caption: str, max_len_text: int) -> str:
    words = caption.split()
    padded_words = words + ["<PAD>"] * (max_len_text - len(words))
    return " ".join(padded_words)
