from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / '.env', override=False)


def _env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    normalized = str(value).strip() if value is not None else ''
    return normalized or default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be one of true/false, 1/0, yes/no, or on/off')


DATA_ROOT = _env_path('REAXIS_DATA_ROOT', REPO_ROOT / 'data')
DATASETS_ROOT = _env_path('REAXIS_DATASETS_ROOT', DATA_ROOT / 'datasets')
SESSIONS_ROOT = _env_path('REAXIS_SESSIONS_ROOT', DATA_ROOT / 'sessions')
UPLOADS_ROOT = _env_path('REAXIS_UPLOADS_ROOT', DATA_ROOT / 'uploads')
OUTPUT_ROOT = _env_path('REAXIS_OUTPUT_ROOT', REPO_ROOT / 'outputs')
RAW_DATASETS_ROOT = _env_path('REAXIS_RAW_DATASETS_ROOT', DATA_ROOT / 'raw')
DATASYNTH_RAW_ROOT = _env_path('REAXIS_DATASYNTH_ROOT', RAW_DATASETS_ROOT)

DEFAULT_DATASET_NAME = _env_text('REAXIS_DEFAULT_DATASET', 'ISIC2017')
if Path(DEFAULT_DATASET_NAME).name != DEFAULT_DATASET_NAME:
    raise ValueError('REAXIS_DEFAULT_DATASET must be a single dataset directory name')
DEFAULT_DATASET_ROOT = (DATASETS_ROOT / DEFAULT_DATASET_NAME).resolve()
LEGACY_AXIS_LIBRARY_PATH = _env_path(
    'REAXIS_LEGACY_AXIS_LIBRARY_PATH',
    DATA_ROOT / 'axis_library.json',
)

BACKEND_HOST = _env_text('REAXIS_BACKEND_HOST', '0.0.0.0')
BACKEND_PORT = _env_int('REAXIS_BACKEND_PORT', 5001)
FLASK_DEBUG = _env_bool('REAXIS_FLASK_DEBUG', True)

LLM_PROVIDER = _env_text('REAXIS_LLM_PROVIDER', 'gemini_api')
GEMINI_MODEL_NAME = _env_text('REAXIS_GEMINI_MODEL', 'gemini-2.5-flash-lite')
GEMINI_API_KEY_PATH = _env_path(
    'REAXIS_GEMINI_API_KEY_FILE',
    DATA_ROOT / 'secrets' / 'gemini_api_key.txt',
)
HF_LOCAL_MODEL_PATH = _env_text('REAXIS_HF_MODEL_PATH', '')
