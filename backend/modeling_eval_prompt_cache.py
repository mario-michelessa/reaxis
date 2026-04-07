#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

try:
    from .llm_inference import LightweightLLMEngine
except ImportError:
    from llm_inference import LightweightLLMEngine


def _utc_now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _normalize_prompt_list(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: List[str] = []
    seen = set()
    for raw in values:
        text = re.sub(r'\s+', ' ', str(raw or '').strip())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _task_value(task: Any, key: str, default: Any = '') -> Any:
    if isinstance(task, Mapping):
        return task.get(key, default)
    return getattr(task, key, default)


def _retry_wait_seconds(error_text: str, default_wait_sec: float) -> float:
    match = re.search(r'retry in\s+([0-9]+(?:\.[0-9]+)?)s', str(error_text or ''), flags=re.I)
    if match:
        try:
            return max(1.0, float(match.group(1)))
        except Exception:
            return float(default_wait_sec)
    return float(default_wait_sec)


def _normalize_record(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    task_id = str(raw.get('task_id') or '').strip()
    dataset_name = str(raw.get('dataset_name') or '').strip()
    query = re.sub(r'\s+', ' ', str(raw.get('query') or '').strip())
    pos_prompts = _normalize_prompt_list(raw.get('pos_prompts'))
    neg_prompts = _normalize_prompt_list(raw.get('neg_prompts'))
    if not task_id or not query or len(pos_prompts) < 2 or len(neg_prompts) < 2:
        return None
    return {
        'task_id': task_id,
        'dataset_name': dataset_name,
        'query': query,
        'pos_prompts': pos_prompts,
        'neg_prompts': neg_prompts,
        'provider': str(raw.get('provider') or 'unknown').strip() or 'unknown',
        'source': str(raw.get('source') or 'unknown').strip() or 'unknown',
        'generated_at_utc': str(raw.get('generated_at_utc') or '').strip(),
    }


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    tasks = payload.get('tasks') if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw in tasks:
        normalized = _normalize_record(raw)
        if not normalized:
            continue
        out[normalized['task_id']] = normalized
    return out


def _write_cache(path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[key] for key in sorted(records)]
    payload = {
        'generated_at_utc': _utc_now_iso(),
        'tasks': ordered,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def hydrate_prompt_cache(
    tasks: Iterable[Any],
    *,
    cache_path: Path,
    source: str,
    n_prompts: int,
    regenerate: bool,
    max_retries: int,
    retry_wait_sec: float,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    normalized_source = str(source or 'template').strip().lower()
    task_list = list(tasks)
    records: Dict[str, Dict[str, Any]] = {} if regenerate else _load_cache(cache_path)

    def log(message: str) -> None:
        if logger is not None:
            logger(str(message))

    if normalized_source != 'llm_cache':
        for task in task_list:
            record = _normalize_record(
                {
                    'task_id': _task_value(task, 'task_id', ''),
                    'dataset_name': _task_value(task, 'dataset_name', ''),
                    'query': _task_value(task, 'query', ''),
                    'pos_prompts': _task_value(task, 'pos_prompts', []),
                    'neg_prompts': _task_value(task, 'neg_prompts', []),
                    'provider': 'fixed_template',
                    'source': 'fixed_template',
                    'generated_at_utc': _utc_now_iso(),
                },
            )
            if record:
                records[record['task_id']] = record
        _write_cache(cache_path, records)
        return records

    llm_engine = LightweightLLMEngine()
    requested_count = max(2, int(n_prompts))
    for index, task in enumerate(task_list, start=1):
        task_id = str(_task_value(task, 'task_id', '')).strip()
        dataset_name = str(_task_value(task, 'dataset_name', '')).strip()
        query = re.sub(r'\s+', ' ', str(_task_value(task, 'query', '')).strip())
        cached = records.get(task_id)
        if cached and cached.get('query') == query and len(cached.get('pos_prompts') or []) >= 2 and len(cached.get('neg_prompts') or []) >= 2:
            continue
        if not task_id or not query:
            raise RuntimeError(f'Invalid modeling-eval task for prompt caching: task_id={task_id!r} query={query!r}')
        error_text = ''
        for attempt in range(1, max(1, int(max_retries)) + 1):
            log(
                f'[modeling-eval] generating contrastive prompts {index}/{len(task_list)} '
                f'task_id={task_id} dataset={dataset_name} attempt={attempt}'
            )
            result = llm_engine.generate_axis_prompt_ensemble(
                attribute=query,
                n_prompts=requested_count,
                dataset_name=dataset_name,
            )
            pos_prompts = _normalize_prompt_list(result.get('pos_prompts'))
            neg_prompts = _normalize_prompt_list(result.get('neg_prompts'))
            if len(pos_prompts) >= 2 and len(neg_prompts) >= 2:
                records[task_id] = {
                    'task_id': task_id,
                    'dataset_name': dataset_name,
                    'query': query,
                    'pos_prompts': pos_prompts[:requested_count],
                    'neg_prompts': neg_prompts[:requested_count],
                    'provider': str(result.get('provider') or 'unknown').strip() or 'unknown',
                    'source': 'llm_cache',
                    'generated_at_utc': _utc_now_iso(),
                }
                _write_cache(cache_path, records)
                break
            error_text = str(result.get('error') or 'LLM prompt generation returned an invalid prompt list').strip()
            if attempt >= int(max_retries):
                raise RuntimeError(
                    f'Failed to generate contrastive prompts for task_id={task_id} query={query!r}: {error_text}'
                )
            wait_sec = _retry_wait_seconds(error_text, retry_wait_sec)
            log(
                f'[modeling-eval] prompt generation retry task_id={task_id} '
                f'waiting={wait_sec:.1f}s error={error_text}'
            )
            time.sleep(wait_sec)
    return records
