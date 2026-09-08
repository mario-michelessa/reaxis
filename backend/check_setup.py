from __future__ import annotations

from pathlib import Path

from .embeddings import embedding_cache_filename
from .runtime_config import (
    DATASETS_ROOT,
    DEFAULT_DATASET_NAME,
    GEMINI_API_KEY_PATH,
    HF_LOCAL_MODEL_PATH,
    LLM_PROVIDER,
    OUTPUT_ROOT,
    RAW_DATASETS_ROOT,
    SESSIONS_ROOT,
)


def _status(label: str, value: object, ok: bool) -> None:
    marker = 'ok' if ok else 'missing'
    print(f'[{marker:7}] {label}: {value}')


def _prepared_datasets(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith('.'))


def main() -> int:
    print('Reaxis configuration')
    _status('prepared datasets root', DATASETS_ROOT, DATASETS_ROOT.is_dir())
    _status('raw datasets root', RAW_DATASETS_ROOT, RAW_DATASETS_ROOT.is_dir())
    _status('sessions root', SESSIONS_ROOT, SESSIONS_ROOT.is_dir())
    _status('output root', OUTPUT_ROOT, OUTPUT_ROOT.is_dir())
    print(f'[config ] default dataset: {DEFAULT_DATASET_NAME}')
    print(f'[config ] LLM provider: {LLM_PROVIDER}')

    if LLM_PROVIDER == 'gemini_api':
        _status('Gemini key file', GEMINI_API_KEY_PATH, GEMINI_API_KEY_PATH.is_file())
    elif LLM_PROVIDER == 'huggingface_local':
        model_path = Path(HF_LOCAL_MODEL_PATH).expanduser() if HF_LOCAL_MODEL_PATH else None
        _status('local HF model', model_path or '<unset>', bool(model_path and model_path.exists()))
    else:
        raise ValueError(f'Unsupported REAXIS_LLM_PROVIDER: {LLM_PROVIDER}')

    datasets = _prepared_datasets(DATASETS_ROOT)
    print(f'\nPrepared datasets: {len(datasets)}')
    required_caches = (
        embedding_cache_filename('clip', normalize=True),
        embedding_cache_filename('clip', normalize=False),
    )
    default_found = False
    default_caches_ok = False
    cache_failures = 0
    for dataset in datasets:
        metadata = dataset / 'metadata.csv'
        cache_status = {
            name: (dataset / '.cache' / name).is_file()
            for name in required_caches
        }
        is_default = dataset.name == DEFAULT_DATASET_NAME
        default_found = default_found or is_default
        if is_default:
            default_caches_ok = all(cache_status.values())
        if not all(cache_status.values()):
            cache_failures += 1
        cache_text = ' '.join(
            f'{name}={"yes" if exists else "no"}'
            for name, exists in cache_status.items()
        )
        print(
            f'- {dataset.name}: metadata={"yes" if metadata.is_file() else "no"} '
            f'{cache_text}'
        )

    if not default_found:
        print(f'\n[missing] default dataset directory: {DATASETS_ROOT / DEFAULT_DATASET_NAME}')
    if not datasets:
        print('\nNo prepared datasets are available. Follow data/README.md before running the UI.')
    elif cache_failures:
        print(
            f'\n[warning] {cache_failures} optional dataset(s) lack one or more UI CLIP caches; '
            'run normalized and --raw precompute before selecting them.'
        )
    if default_found and not default_caches_ok:
        print(f'[missing] default dataset lacks a required UI cache: {DEFAULT_DATASET_NAME}')
    return 0 if datasets and default_found and default_caches_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
