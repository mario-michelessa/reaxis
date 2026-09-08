#!/usr/bin/env python3
from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    from .constants import (
        AXIS_BUILDER_LLM_PROMPT_COUNT,
        AXIS_BUILDER_PROMPT_ENSEMBLE_SYSTEM_PROMPT,
        AXIS_BUILDER_PROMPT_ENSEMBLE_USER_PROMPT_TEMPLATE,
        DATASET_LLM_CONTEXT,
        ATTRIBUTE_EXTRACTION_SYSTEM_PROMPT,
        ATTRIBUTE_SUPPORT_SYSTEM_PROMPT,
        ATTRIBUTE_SUPPORT_USER_PROMPT_TEMPLATE,
        ATTRIBUTE_EXTRACTION_USER_PROMPT_TEMPLATE,
        ATTRIBUTE_VALUE_SUGGESTION_SYSTEM_PROMPT,
        ATTRIBUTE_VALUE_SUGGESTION_USER_PROMPT_TEMPLATE,
        ATTRIBUTE_CONTINUOUS_ANCHORS_SYSTEM_PROMPT,
        ATTRIBUTE_CONTINUOUS_ANCHORS_USER_PROMPT_TEMPLATE,
        DEFAULT_MAX_ATTRIBUTES,
        DEFAULT_VALUE_COUNT,
        GEMINI_API_KEY_PATH,
        GEMINI_API_TIMEOUT_SEC,
        GEMINI_MODEL_NAME,
        HF_4BIT_COMPUTE_DTYPE,
        HF_4BIT_DEVICE_MAP,
        HF_4BIT_QUANT_TYPE,
        HF_4BIT_USE_DOUBLE_QUANT,
        HF_LOCAL_FILES_ONLY,
        HF_LOCAL_MODEL_PATH,
        HF_MAX_NEW_TOKENS,
        HF_TEMPERATURE,
        HF_TOP_P,
        HF_TRUST_REMOTE_CODE,
        HF_USE_4BIT,
        LLM_PROVIDER,
        MAX_VALUE_COUNT,
    )
except ImportError:
    from constants import (
        AXIS_BUILDER_LLM_PROMPT_COUNT,
        AXIS_BUILDER_PROMPT_ENSEMBLE_SYSTEM_PROMPT,
        AXIS_BUILDER_PROMPT_ENSEMBLE_USER_PROMPT_TEMPLATE,
        DATASET_LLM_CONTEXT,
        ATTRIBUTE_EXTRACTION_SYSTEM_PROMPT,
        ATTRIBUTE_SUPPORT_SYSTEM_PROMPT,
        ATTRIBUTE_SUPPORT_USER_PROMPT_TEMPLATE,
        ATTRIBUTE_EXTRACTION_USER_PROMPT_TEMPLATE,
        ATTRIBUTE_VALUE_SUGGESTION_SYSTEM_PROMPT,
        ATTRIBUTE_VALUE_SUGGESTION_USER_PROMPT_TEMPLATE,
        ATTRIBUTE_CONTINUOUS_ANCHORS_SYSTEM_PROMPT,
        ATTRIBUTE_CONTINUOUS_ANCHORS_USER_PROMPT_TEMPLATE,
        DEFAULT_MAX_ATTRIBUTES,
        DEFAULT_VALUE_COUNT,
        GEMINI_API_KEY_PATH,
        GEMINI_API_TIMEOUT_SEC,
        GEMINI_MODEL_NAME,
        HF_4BIT_COMPUTE_DTYPE,
        HF_4BIT_DEVICE_MAP,
        HF_4BIT_QUANT_TYPE,
        HF_4BIT_USE_DOUBLE_QUANT,
        HF_LOCAL_FILES_ONLY,
        HF_LOCAL_MODEL_PATH,
        HF_MAX_NEW_TOKENS,
        HF_TEMPERATURE,
        HF_TOP_P,
        HF_TRUST_REMOTE_CODE,
        HF_USE_4BIT,
        LLM_PROVIDER,
        MAX_VALUE_COUNT,
    )


VALID_ATTRIBUTE_TYPES = {'categorical', 'ordinal', 'continuous'}


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        v = str(raw or '').strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(n)))


def _resample_ordered(values: List[str], n: int) -> List[str]:
    values = _dedupe_keep_order(values)
    if len(values) < 2:
        return []
    n = _clamp(n, 2, MAX_VALUE_COUNT)
    if len(values) <= n:
        return values
    # Downsample while preserving order and endpoints.
    idxs = []
    for i in range(n):
        idx = round(i * (len(values) - 1) / (n - 1))
        idxs.append(idx)
    return [values[i] for i in idxs]


def _truncate_keep_order(values: List[str], n: int) -> List[str]:
    vals = _dedupe_keep_order(values)
    n = _clamp(n, 2, MAX_VALUE_COUNT)
    if len(vals) <= n:
        return vals
    return vals[:n]


def _normalize_attribute_type(v: Any) -> Optional[str]:
    s = str(v or '').strip().lower()
    if not s:
        return None
    if s in VALID_ATTRIBUTE_TYPES:
        return s
    if s in {'nominal', 'class', 'classes', 'discrete', 'category', 'categories'}:
        return 'categorical'
    if s in {'ordered', 'ranked', 'rank', 'ranking'}:
        return 'ordinal'
    if s in {'numeric', 'number', 'scalar', 'real', 'continuous-scale'}:
        return 'continuous'
    if 'categor' in s:
        return 'categorical'
    if 'ordin' in s or 'rank' in s:
        return 'ordinal'
    if 'contin' in s or 'numeric' in s or 'scalar' in s:
        return 'continuous'
    return None


def _extract_name_and_type(x: Any) -> Dict[str, Optional[str]]:
    if isinstance(x, dict):
        name = str(
            x.get('name')
            or x.get('attribute')
            or x.get('label')
            or x.get('key')
            or ''
        ).strip()
        atype = _normalize_attribute_type(
            x.get('type')
            or x.get('attribute_type')
            or x.get('kind')
        )
        return {'name': name or None, 'type': atype}
    name = str(x or '').strip()
    return {'name': name or None, 'type': None}


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or '').strip()
    if not raw:
        return None

    decoder = json.JSONDecoder()
    candidates: List[str] = [raw]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.I | re.S)
    for block in fenced:
        text_block = str(block or '').strip()
        if text_block:
            candidates.append(text_block)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

        for match in re.finditer(r'\{', candidate):
            start = match.start()
            try:
                obj, end = decoder.raw_decode(candidate[start:])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _resolve_torch_dtype(name: str, torch_mod):
    key = str(name or '').strip().lower()
    if key in {'float16', 'fp16', 'half'}:
        return torch_mod.float16
    if key in {'bfloat16', 'bf16'}:
        return torch_mod.bfloat16
    if key in {'float32', 'fp32'}:
        return torch_mod.float32
    return torch_mod.float16


def _short(text: str, n: int = 140) -> str:
    s = re.sub(r'\s+', ' ', str(text or '').strip())
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + '...'


def _dataset_prompt_context(dataset_name: Any) -> Dict[str, str]:
    raw_name = str(dataset_name or '').strip()
    canonical_name = Path(raw_name).name if raw_name else ''
    context = re.sub(r'\s+', ' ', str(DATASET_LLM_CONTEXT.get(canonical_name) or '').strip()).strip()
    return {
        'dataset_name': canonical_name or 'unspecified dataset',
        'dataset_context': context or 'No dataset-specific context provided.',
    }


def _normalize_name_key(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '').strip().lower())


def _clamp_score(value: Any, default: float) -> float:
    try:
        score = float(value)
    except Exception:
        return float(default)
    return float(max(0.0, min(1.0, score)))


def _clean_support_text(value: Any) -> str:
    text = str(value or '').strip().strip('"').strip("'")
    return re.sub(r'\s+', ' ', text).strip()


def _token_spans(text: str) -> List[Dict[str, Any]]:
    return [
        {
            'text': m.group(0),
            'start': int(m.start()),
            'end': int(m.end()),
        }
        for m in re.finditer(r'\S+', str(text or ''))
    ]


def _overlaps_existing(start: int, end: int, used_ranges: List[tuple[int, int]]) -> bool:
    for lo, hi in used_ranges:
        if start < hi and end > lo:
            return True
    return False


def _align_phrase_to_prompt(prompt: str, phrase: str, used_ranges: Optional[List[tuple[int, int]]] = None):
    source = str(prompt or '')
    target = _clean_support_text(phrase)
    if not source or not target:
        return None

    used = used_ranges if isinstance(used_ranges, list) else []
    src_lower = source.lower()
    tgt_lower = target.lower()

    matches = list(re.finditer(re.escape(tgt_lower), src_lower))
    for m in matches:
        start = int(m.start())
        end = int(m.end())
        if _overlaps_existing(start, end, used):
            continue
        used.append((start, end))
        return {
            'start': start,
            'end': end,
            'text': source[start:end],
            'match_score': 1.0,
        }

    pattern = r'\b' + r'\s+'.join(re.escape(part) for part in re.split(r'\s+', tgt_lower) if part) + r'\b'
    if pattern != r'\b\b':
        for m in re.finditer(pattern, src_lower, flags=re.S):
            start = int(m.start())
            end = int(m.end())
            if _overlaps_existing(start, end, used):
                continue
            used.append((start, end))
            return {
                'start': start,
                'end': end,
                'text': source[start:end],
                'match_score': 0.94,
            }

    prompt_tokens = _token_spans(source)
    phrase_tokens = [tok.lower() for tok in re.findall(r'\S+', tgt_lower)]
    if len(prompt_tokens) == 0 or len(phrase_tokens) == 0:
        return None

    target_joined = ' '.join(phrase_tokens)
    best = None
    target_len = len(phrase_tokens)
    min_window = max(1, target_len - 2)
    max_window = min(len(prompt_tokens), target_len + 2)
    for window_size in range(min_window, max_window + 1):
        for i in range(0, len(prompt_tokens) - window_size + 1):
            start = int(prompt_tokens[i]['start'])
            end = int(prompt_tokens[i + window_size - 1]['end'])
            if _overlaps_existing(start, end, used):
                continue
            candidate = source[start:end]
            candidate_joined = ' '.join(tok.lower() for tok in re.findall(r'\S+', candidate))
            ratio = SequenceMatcher(None, target_joined, candidate_joined).ratio()
            if best is None or ratio > best['match_score']:
                best = {
                    'start': start,
                    'end': end,
                    'text': candidate,
                    'match_score': float(ratio),
                }
    if best and best['match_score'] >= 0.72:
        used.append((best['start'], best['end']))
        return best
    return None


class LightweightLLMEngine:
    def __init__(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.provider = LLM_PROVIDER
        self.gemini_model_name = str(GEMINI_MODEL_NAME or 'gemini-2.5-flash-lite').strip()
        gemini_key_path = Path(str(GEMINI_API_KEY_PATH or 'data/secrets/gemini_api_key.txt')).expanduser()
        if not gemini_key_path.is_absolute():
            gemini_key_path = repo_root / gemini_key_path
        self.gemini_api_key_path = gemini_key_path.resolve()
        self.gemini_api_timeout_sec = max(1, int(GEMINI_API_TIMEOUT_SEC or 45))
        self.hf_local_model_path = HF_LOCAL_MODEL_PATH
        self.hf_local_files_only = bool(HF_LOCAL_FILES_ONLY)
        self.hf_trust_remote_code = bool(HF_TRUST_REMOTE_CODE)
        self.hf_max_new_tokens = int(HF_MAX_NEW_TOKENS)
        self.hf_temperature = float(HF_TEMPERATURE)
        self.hf_top_p = float(HF_TOP_P)
        self.hf_use_4bit = bool(HF_USE_4BIT)
        self.hf_4bit_quant_type = str(HF_4BIT_QUANT_TYPE or 'nf4')
        self.hf_4bit_compute_dtype = str(HF_4BIT_COMPUTE_DTYPE or 'float16')
        self.hf_4bit_use_double_quant = bool(HF_4BIT_USE_DOUBLE_QUANT)
        self.hf_4bit_device_map = str(HF_4BIT_DEVICE_MAP or 'auto')
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_load_error: Optional[str] = None
        self._gemini_api_key: Optional[str] = None
        self._gemini_api_key_error: Optional[str] = None
        self._last_runtime_error: Optional[str] = None
        self._axis_prompt_ensemble_cache: Dict[tuple[str, str, int], Dict[str, Any]] = {}
        self._log(
            'init provider=%s gemini_model=%s gemini_key_path=%s model_path=%s use_4bit=%s quant_type=%s compute_dtype=%s device_map=%s',
            self.provider,
            self.gemini_model_name,
            self.gemini_api_key_path,
            self.hf_local_model_path,
            self.hf_use_4bit,
            self.hf_4bit_quant_type,
            self.hf_4bit_compute_dtype,
            self.hf_4bit_device_map,
        )

    def _log(self, msg: str, *args):
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f'{msg} {args}'
        print(f'[llm] {msg}')

    def _set_last_runtime_error(self, message: Optional[str]) -> None:
        text = str(message or '').strip()
        self._last_runtime_error = text or None

    def _can_use_hf_local(self) -> bool:
        return self.provider in {'huggingface_local', 'huggingface', 'hf_local'} and bool(self.hf_local_model_path)

    def _can_use_gemini_api(self) -> bool:
        return self.provider in {'gemini_api', 'gemini'}

    def _load_gemini_api_key(self) -> Optional[str]:
        if not self._can_use_gemini_api():
            return None
        if self._gemini_api_key:
            return self._gemini_api_key
        if self._gemini_api_key_error:
            self._log('skip gemini key load: previous error=%s', self._gemini_api_key_error)
            return None
        try:
            raw = self.gemini_api_key_path.read_text(encoding='utf-8').strip()
            if not raw:
                raise RuntimeError(f'Empty Gemini API key file: {self.gemini_api_key_path}')
            self._gemini_api_key = raw
            self._log('gemini api key loaded from %s', self.gemini_api_key_path)
            return self._gemini_api_key
        except Exception as e:
            self._gemini_api_key_error = str(e)
            self._log('gemini api key load failed: %s', self._gemini_api_key_error)
            return None

    def _load_hf_local(self) -> bool:
        if not self._can_use_hf_local():
            self._log('skip load: provider/path invalid provider=%s path=%s', self.provider, self.hf_local_model_path)
            return False
        if self._hf_model is not None and self._hf_tokenizer is not None:
            self._log('reuse loaded HF model')
            return True
        if self._hf_load_error:
            self._log('skip load: previous error=%s', self._hf_load_error)
            return False
        try:
            t0 = time.time()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self._log('loading tokenizer from %s', self.hf_local_model_path)
            tok = AutoTokenizer.from_pretrained(
                self.hf_local_model_path,
                local_files_only=self.hf_local_files_only,
                trust_remote_code=self.hf_trust_remote_code,
            )
            self._log('tokenizer loaded')
            model_kwargs: Dict[str, Any] = {
                'local_files_only': self.hf_local_files_only,
                'trust_remote_code': self.hf_trust_remote_code,
            }
            if self.hf_use_4bit:
                if not torch.cuda.is_available():
                    raise RuntimeError('HF_USE_4BIT=True requires a CUDA-capable GPU with bitsandbytes support')
                self._log(
                    'loading model in 4-bit (quant_type=%s compute_dtype=%s double_quant=%s device_map=%s)',
                    self.hf_4bit_quant_type,
                    self.hf_4bit_compute_dtype,
                    self.hf_4bit_use_double_quant,
                    self.hf_4bit_device_map,
                )
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.hf_4bit_quant_type,
                    bnb_4bit_compute_dtype=_resolve_torch_dtype(self.hf_4bit_compute_dtype, torch),
                    bnb_4bit_use_double_quant=self.hf_4bit_use_double_quant,
                )
                model_kwargs['quantization_config'] = quant_cfg
                model_kwargs['device_map'] = self.hf_4bit_device_map
            elif torch.cuda.is_available():
                self._log('loading model in fp16 on cuda')
                model_kwargs['torch_dtype'] = torch.float16
            else:
                self._log('loading model in default precision on cpu')

            model = AutoModelForCausalLM.from_pretrained(
                self.hf_local_model_path,
                **model_kwargs,
            )
            if not self.hf_use_4bit:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                self._log('model moved to device=%s', device)
            model.eval()
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            self._hf_model = model
            self._hf_tokenizer = tok
            self._log('model load complete in %.2fs', time.time() - t0)
            return True
        except Exception as e:
            self._hf_load_error = str(e)
            self._log('model load failed: %s', self._hf_load_error)
            return False

    def _render_messages_prompt(self, messages: List[Dict[str, str]]) -> str:
        tok = self._hf_tokenizer
        if tok is not None and hasattr(tok, 'apply_chat_template'):
            try:
                rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception as exc:
                self._log('chat template rendering failed; using plain prompt: %s', exc)
        chunks: List[str] = []
        for m in messages:
            role = str(m.get('role') or 'user').strip().capitalize()
            content = str(m.get('content') or '').strip()
            chunks.append(f'{role}:\n{content}')
        chunks.append('Assistant:\n')
        return '\n\n'.join(chunks)

    def _post_hf_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Optional[str]:
        if not self._load_hf_local():
            self._log('chat aborted: model not loaded')
            self._set_last_runtime_error('HF local model is not loaded')
            return None
        try:
            t0 = time.time()
            import torch

            tok = self._hf_tokenizer
            model = self._hf_model
            if tok is None or model is None:
                self._log('chat aborted: tokenizer/model missing')
                self._set_last_runtime_error('HF tokenizer/model missing')
                return None

            self._log('chat start messages=%d temperature=%.3f', len(messages), float(temperature))
            prompt = self._render_messages_prompt(messages)
            self._log('rendered prompt chars=%d preview="%s"', len(prompt), _short(prompt))
            inputs = tok(prompt, return_tensors='pt')
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            self._log('tokenized input_tokens=%d device=%s', int(inputs['input_ids'].shape[1]), model_device)

            temp = max(0.0, float(temperature))
            do_sample = temp > 1e-6
            gen_kwargs: Dict[str, Any] = {
                'max_new_tokens': int(self.hf_max_new_tokens),
                'do_sample': do_sample,
                'pad_token_id': tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                'eos_token_id': tok.eos_token_id,
            }
            if do_sample:
                gen_kwargs['temperature'] = max(temp, 1e-5)
                gen_kwargs['top_p'] = float(self.hf_top_p)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            input_len = int(inputs['input_ids'].shape[1])
            new_tokens = out[0][input_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            self._log(
                'chat complete new_tokens=%d output_chars=%d elapsed=%.2fs',
                int(new_tokens.shape[0]),
                len(str(text or '')),
                time.time() - t0,
            )
            return str(text or '').strip()
        except Exception as e:
            self._log('chat failed: %s', e)
            self._set_last_runtime_error(f'HF chat failed: {e}')
            return None

    def _post_gemini_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Optional[str]:
        api_key = self._load_gemini_api_key()
        if not api_key:
            self._log('gemini chat aborted: api key unavailable')
            self._set_last_runtime_error('Gemini API key unavailable')
            return None
        try:
            t0 = time.time()
            system_parts = [str(m.get('content') or '').strip() for m in messages if str(m.get('role') or '').strip().lower() == 'system']
            non_system = [m for m in messages if str(m.get('role') or '').strip().lower() != 'system']
            contents = []
            for m in non_system:
                role = str(m.get('role') or 'user').strip().lower()
                text = str(m.get('content') or '').strip()
                if not text:
                    continue
                mapped_role = 'model' if role == 'assistant' else 'user'
                contents.append({
                    'role': mapped_role,
                    'parts': [{'text': text}],
                })
            if len(contents) == 0:
                self._log('gemini chat aborted: no message contents')
                self._set_last_runtime_error('Gemini chat aborted: no message contents')
                return None
            payload: Dict[str, Any] = {
                'contents': contents,
                'generationConfig': {
                    'temperature': max(0.0, float(temperature)),
                    'responseMimeType': 'application/json',
                },
            }
            if system_parts:
                payload['systemInstruction'] = {
                    'parts': [{'text': '\n\n'.join(system_parts)}],
                }
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_name}:generateContent'
            body = json.dumps(payload).encode('utf-8')
            req = urllib_request.Request(
                url,
                data=body,
                method='POST',
                headers={
                    'Content-Type': 'application/json',
                    'X-goog-api-key': api_key,
                },
            )
            self._log('gemini chat start messages=%d temperature=%.3f model=%s', len(messages), float(temperature), self.gemini_model_name)
            with urllib_request.urlopen(req, timeout=self.gemini_api_timeout_sec) as resp:
                raw = resp.read().decode('utf-8', errors='replace')
            obj = json.loads(raw)
            candidates = obj.get('candidates') if isinstance(obj, dict) else None
            if not isinstance(candidates, list) or len(candidates) == 0:
                self._log('gemini chat failed: missing candidates raw="%s"', _short(raw))
                self._set_last_runtime_error(f'Gemini API returned no candidates: {_short(raw)}')
                return None
            parts = (((candidates[0] or {}).get('content') or {}).get('parts') or [])
            text_chunks = []
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict):
                        text = str(part.get('text') or '').strip()
                        if text:
                            text_chunks.append(text)
            text = '\n'.join(text_chunks).strip()
            self._log('gemini chat complete output_chars=%d elapsed=%.2fs', len(text), time.time() - t0)
            if not text:
                self._set_last_runtime_error('Gemini API returned empty content')
            return text or None
        except urllib_error.HTTPError as e:
            try:
                details = e.read().decode('utf-8', errors='replace')
            except Exception:
                details = str(e)
            self._log('gemini chat http error: %s body="%s"', e, _short(details))
            summary = None
            try:
                payload = json.loads(details)
                error_obj = payload.get('error') if isinstance(payload, dict) else None
                if isinstance(error_obj, dict):
                    code = error_obj.get('code') or getattr(e, 'code', 'unknown')
                    message = str(error_obj.get('message') or '').strip()
                    if message:
                        summary = f'Gemini API HTTP {code}: {message}'
            except Exception:
                summary = None
            self._set_last_runtime_error(summary or f'Gemini API HTTP {getattr(e, "code", "unknown")}: {_short(details)}')
            return None
        except Exception as e:
            self._log('gemini chat failed: %s', e)
            self._set_last_runtime_error(f'Gemini chat failed: {e}')
            return None

    def _run_structured_json(self, messages: List[Dict[str, str]], temperature: float = 0.2):
        self._set_last_runtime_error(None)
        if self._can_use_gemini_api():
            content = self._post_gemini_chat(messages, temperature=temperature if temperature is not None else self.hf_temperature)
            if content:
                obj = _extract_json_payload(content)
                if obj is not None:
                    self._log('json parse success keys=%s', list(obj.keys()))
                    return obj, 'gemini_api'
                self._log('json parse failed preview="%s"', _short(content))
                self._set_last_runtime_error(f'Gemini API returned non-JSON content: {_short(content)}')
            else:
                self._log('no content returned from gemini')
        if self._can_use_hf_local():
            content = self._post_hf_chat(messages, temperature=temperature if temperature is not None else self.hf_temperature)
            if content:
                obj = _extract_json_payload(content)
                if obj is not None:
                    self._log('json parse success keys=%s', list(obj.keys()))
                    return obj, 'huggingface_local'
                self._log('json parse failed preview="%s"', _short(content))
                self._set_last_runtime_error(f'HF model returned non-JSON content: {_short(content)}')
            else:
                self._log('no content returned from chat')
        return None, None

    def _last_error(self) -> str:
        if self._can_use_gemini_api() and self._gemini_api_key_error:
            return f'Gemini API key load failed: {self._gemini_api_key_error}'
        if self._hf_load_error:
            return f'HF model load failed: {self._hf_load_error}'
        if self._last_runtime_error:
            return self._last_runtime_error
        if self._can_use_gemini_api():
            return 'Gemini API returned empty or non-JSON output'
        return 'HF model returned empty or non-JSON output'

    def extract_attribute_support(self, prompt: str, attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
        p = str(prompt or '').strip()
        attrs = []
        for raw in attributes or []:
            item = _extract_name_and_type(raw)
            name = str(item.get('name') or '').strip()
            atype = _normalize_attribute_type(item.get('type'))
            if not name:
                continue
            attrs.append({'name': name, 'type': atype or ''})
        if not p or len(attrs) == 0:
            return {'supports_by_name': {}, 'provider': 'none'}

        self._log(
            'extract_attribute_support start prompt_len=%d attributes=%s',
            len(p),
            [f"{a.get('name')}:{a.get('type') or 'unknown'}" for a in attrs],
        )
        user_prompt = ATTRIBUTE_SUPPORT_USER_PROMPT_TEMPLATE.format(
            prompt=p,
            attributes_json=json.dumps(attrs, ensure_ascii=True),
        )
        obj, provider = self._run_structured_json([
            {'role': 'system', 'content': ATTRIBUTE_SUPPORT_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ], temperature=self.hf_temperature)

        raw_supports = None
        if isinstance(obj, dict):
            if isinstance(obj.get('supports'), list):
                raw_supports = obj.get('supports')
            elif isinstance(obj.get('attributes'), list):
                raw_supports = obj.get('attributes')
            elif isinstance(obj.get('items'), list):
                raw_supports = obj.get('items')

        supports_by_name: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_supports, list):
            for entry in raw_supports:
                if not isinstance(entry, dict):
                    continue
                name = str(
                    entry.get('name')
                    or entry.get('attribute')
                    or entry.get('label')
                    or ''
                ).strip()
                if not name:
                    continue
                name_key = _normalize_name_key(name)
                raw_spans = entry.get('spans')
                if not isinstance(raw_spans, list):
                    raw_spans = entry.get('support')
                if not isinstance(raw_spans, list):
                    raw_spans = entry.get('phrases')
                if not isinstance(raw_spans, list):
                    raw_spans = []

                used_ranges: List[tuple[int, int]] = []
                support_spans = []
                support_phrases = []
                for idx, raw_span in enumerate(raw_spans[:4]):
                    if isinstance(raw_span, dict):
                        phrase_text = _clean_support_text(
                            raw_span.get('text')
                            or raw_span.get('phrase')
                            or raw_span.get('value')
                        )
                        score = _clamp_score(raw_span.get('score'), 1.0 - (idx * 0.18))
                    else:
                        phrase_text = _clean_support_text(raw_span)
                        score = _clamp_score(None, 1.0 - (idx * 0.18))
                    if not phrase_text:
                        continue
                    aligned = _align_phrase_to_prompt(p, phrase_text, used_ranges=used_ranges)
                    if not aligned:
                        continue
                    support_phrases.append(aligned['text'])
                    support_spans.append({
                        'text': aligned['text'],
                        'start': int(aligned['start']),
                        'end': int(aligned['end']),
                        'score': float(max(0.0, min(1.0, score * float(aligned.get('match_score') or 1.0)))),
                    })

                support_spans.sort(key=lambda row: (int(row.get('start') or 0), -(int(row.get('end') or 0))))
                if support_spans:
                    supports_by_name[name_key] = {
                        'name': name,
                        'support_phrases': support_phrases,
                        'support_spans': support_spans,
                    }

        self._log(
            'extract_attribute_support done provider=%s explained=%d',
            provider or 'unknown',
            len(supports_by_name),
        )
        return {
            'supports_by_name': supports_by_name,
            'provider': provider or 'unknown',
        }

    def extract_attributes(
        self,
        prompt: str,
        max_attributes: int = DEFAULT_MAX_ATTRIBUTES,
        dataset_name: str = '',
    ) -> Dict[str, Any]:
        p = str(prompt or '').strip()
        max_attributes = _clamp(max_attributes, 1, 20)
        dataset_prompt_ctx = _dataset_prompt_context(dataset_name)
        self._log(
            'extract_attributes start dataset=%s prompt_len=%d max_attributes=%d preview="%s"',
            dataset_prompt_ctx['dataset_name'],
            len(p),
            max_attributes,
            _short(p),
        )
        if not p:
            return {'attributes': [], 'provider': 'none'}

        user_prompt = ATTRIBUTE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            prompt=p,
            dataset_name=dataset_prompt_ctx['dataset_name'],
            dataset_context=dataset_prompt_ctx['dataset_context'],
        )
        obj, provider = self._run_structured_json([
            {'role': 'system', 'content': ATTRIBUTE_EXTRACTION_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ], temperature=self.hf_temperature)

        raw_attrs = obj.get('attributes') if isinstance(obj, dict) else None
        if isinstance(raw_attrs, list):
            dedup: List[Dict[str, Any]] = []
            seen = set()
            for raw in raw_attrs:
                item = _extract_name_and_type(raw)
                name = str(item.get('name') or '').strip()
                if not name:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                dedup.append({'name': name, 'type': item.get('type')})
                if len(dedup) >= max_attributes:
                    break

            for item in dedup:
                item['type'] = _normalize_attribute_type(item.get('type'))

            attrs = [a for a in dedup if a.get('name') and a.get('type') in VALID_ATTRIBUTE_TYPES]
            if attrs:
                attrs = attrs[:max_attributes]
                support_result = self.extract_attribute_support(p, attrs)
                supports_by_name = support_result.get('supports_by_name') or {}
                support_provider = support_result.get('provider') or 'unknown'
                enriched_attrs = []
                for attr in attrs:
                    name_key = _normalize_name_key(attr.get('name'))
                    support = supports_by_name.get(name_key) or {}
                    enriched_attrs.append({
                        **attr,
                        'support_phrases': list(support.get('support_phrases') or []),
                        'support_spans': list(support.get('support_spans') or []),
                    })
                attrs = enriched_attrs
                attr_names = [str(a.get('name')) for a in attrs]
                self._log(
                    'extract_attributes success count=%d attrs=%s',
                    len(attrs),
                    [f"{a.get('name')}:{a.get('type')}" for a in attrs],
                )
                return {
                    'attributes': attrs,
                    'attribute_names': attr_names,
                    'provider': provider or 'unknown',
                    'support_provider': support_provider,
                }

        self._log('extract_attributes failed: %s', self._last_error())
        return {
            'attributes': [],
            'attribute_names': [],
            'provider': provider or self.provider or 'unknown',
            'error': self._last_error(),
        }

    def generate_axis_prompt_ensemble(
        self,
        attribute: str,
        n_prompts: int = AXIS_BUILDER_LLM_PROMPT_COUNT,
        dataset_name: str = '',
    ) -> Dict[str, Any]:
        attr = str(attribute or '').strip()
        n_prompts = _clamp(n_prompts, 2, 16)
        dataset_prompt_ctx = _dataset_prompt_context(dataset_name)
        cache_key = (dataset_prompt_ctx['dataset_name'].lower(), attr.lower(), int(n_prompts))
        self._log(
            'generate_axis_prompt_ensemble start dataset=%s attribute="%s" n_prompts=%d',
            dataset_prompt_ctx['dataset_name'],
            attr,
            n_prompts,
        )
        if not attr:
            return {'pos_prompts': [], 'neg_prompts': [], 'provider': 'none'}
        cached = self._axis_prompt_ensemble_cache.get(cache_key)
        if isinstance(cached, dict):
            self._log(
                'generate_axis_prompt_ensemble cache hit dataset=%s attribute="%s" n_prompts=%d provider=%s',
                dataset_prompt_ctx['dataset_name'],
                attr,
                n_prompts,
                str(cached.get('provider') or 'unknown'),
            )
            return {
                'pos_prompts': list(cached.get('pos_prompts') or []),
                'neg_prompts': list(cached.get('neg_prompts') or []),
                'provider': str(cached.get('provider') or 'unknown'),
            }

        user_prompt = AXIS_BUILDER_PROMPT_ENSEMBLE_USER_PROMPT_TEMPLATE.format(
            attribute=attr,
            n_prompts=n_prompts,
            dataset_name=dataset_prompt_ctx['dataset_name'],
            dataset_context=dataset_prompt_ctx['dataset_context'],
        )
        obj, provider = self._run_structured_json([
            {'role': 'system', 'content': AXIS_BUILDER_PROMPT_ENSEMBLE_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ], temperature=max(self.hf_temperature, 0.45))

        raw_pos = None
        raw_neg = None
        if isinstance(obj, dict):
            if isinstance(obj.get('pos_prompts'), list):
                raw_pos = obj.get('pos_prompts')
            elif isinstance(obj.get('positive_prompts'), list):
                raw_pos = obj.get('positive_prompts')
            if isinstance(obj.get('neg_prompts'), list):
                raw_neg = obj.get('neg_prompts')
            elif isinstance(obj.get('negative_prompts'), list):
                raw_neg = obj.get('negative_prompts')
        if isinstance(raw_pos, list) and isinstance(raw_neg, list):
            pos_prompts = _dedupe_keep_order(
                [str(v).strip() for v in raw_pos if str(v).strip()]
            )[:n_prompts]
            neg_prompts = _dedupe_keep_order(
                [str(v).strip() for v in raw_neg if str(v).strip()]
            )[:n_prompts]
            if len(pos_prompts) >= 2 and len(neg_prompts) >= 2:
                self._log(
                    'generate_axis_prompt_ensemble success pos_count=%d neg_count=%d pos=%s neg=%s',
                    len(pos_prompts),
                    len(neg_prompts),
                    pos_prompts,
                    neg_prompts,
                )
                self._axis_prompt_ensemble_cache[cache_key] = {
                    'pos_prompts': list(pos_prompts),
                    'neg_prompts': list(neg_prompts),
                    'provider': provider or 'unknown',
                }
                return {
                    'pos_prompts': pos_prompts,
                    'neg_prompts': neg_prompts,
                    'provider': provider or 'unknown',
                }

        self._log('generate_axis_prompt_ensemble failed: %s', self._last_error())
        return {
            'pos_prompts': [],
            'neg_prompts': [],
            'provider': provider or self.provider or 'unknown',
            'error': self._last_error(),
        }

    def suggest_categories(
        self,
        attribute: str,
        attribute_type: str,
        n_values: int = DEFAULT_VALUE_COUNT,
        provided_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        attr = str(attribute or '').strip()
        atype = _normalize_attribute_type(attribute_type)
        n_values = _clamp(n_values, 2, MAX_VALUE_COUNT)
        self._log(
            'suggest_categories start attribute="%s" type=%s n_values=%d provided_categories=%d',
            attr,
            atype,
            n_values,
            len(provided_categories) if isinstance(provided_categories, list) else 0,
        )
        if not attr:
            return {'categories': [], 'provider': 'none'}
        if atype not in {'categorical', 'ordinal'}:
            return {'categories': [], 'provider': 'none', 'error': f'Invalid attribute_type: {attribute_type}'}

        if isinstance(provided_categories, list) and len(provided_categories) > 0:
            if atype == 'ordinal':
                vals = _resample_ordered([str(v) for v in provided_categories], n_values)
            else:
                vals = _truncate_keep_order([str(v) for v in provided_categories], n_values)
            if len(vals) >= 2:
                self._log('suggest_categories using provided categories count=%d', len(vals))
                return {'categories': vals, 'provider': 'provided'}

        user_prompt = ATTRIBUTE_VALUE_SUGGESTION_USER_PROMPT_TEMPLATE.format(
            attribute=attr,
            attribute_type=atype,
            n_values=n_values,
        )
        obj, provider = self._run_structured_json([
            {'role': 'system', 'content': ATTRIBUTE_VALUE_SUGGESTION_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ], temperature=self.hf_temperature)

        raw_vals = None
        if isinstance(obj, dict):
            if isinstance(obj.get('categories'), list):
                raw_vals = obj.get('categories')
            elif isinstance(obj.get('values'), list):
                raw_vals = obj.get('values')
        if isinstance(raw_vals, list):
            if atype == 'ordinal':
                vals = _resample_ordered([str(v).strip() for v in raw_vals if str(v).strip()], n_values)
            else:
                vals = _truncate_keep_order([str(v).strip() for v in raw_vals if str(v).strip()], n_values)
            if len(vals) >= 2:
                self._log('suggest_categories success count=%d categories=%s', len(vals), vals)
                return {'categories': vals, 'provider': provider or 'unknown'}

        self._log('suggest_categories failed: %s', self._last_error())
        return {
            'categories': [],
            'provider': provider or self.provider or 'unknown',
            'error': self._last_error(),
        }

    def suggest_continuous_anchors(
        self,
        attribute: str,
        provided_low: Optional[str] = None,
        provided_high: Optional[str] = None,
    ) -> Dict[str, Any]:
        attr = str(attribute or '').strip()
        low_given = str(provided_low or '').strip()
        high_given = str(provided_high or '').strip()
        self._log(
            'suggest_continuous_anchors start attribute="%s" provided_low=%s provided_high=%s',
            attr,
            bool(low_given),
            bool(high_given),
        )
        if not attr:
            return {'low': '', 'high': '', 'provider': 'none'}

        if low_given and high_given and low_given.lower() != high_given.lower():
            self._log('suggest_continuous_anchors using provided anchors')
            return {
                'low': low_given,
                'high': high_given,
                'provider': 'provided',
            }

        user_prompt = ATTRIBUTE_CONTINUOUS_ANCHORS_USER_PROMPT_TEMPLATE.format(
            attribute=attr,
        )
        obj, provider = self._run_structured_json([
            {'role': 'system', 'content': ATTRIBUTE_CONTINUOUS_ANCHORS_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ], temperature=self.hf_temperature)

        low = ''
        high = ''
        if isinstance(obj, dict):
            low = str(
                obj.get('low')
                or obj.get('min')
                or obj.get('negative')
                or obj.get('anchor_low')
                or ''
            ).strip()
            high = str(
                obj.get('high')
                or obj.get('max')
                or obj.get('positive')
                or obj.get('anchor_high')
                or ''
            ).strip()
            if (not low or not high) and isinstance(obj.get('anchors'), list):
                anchors = [str(v).strip() for v in obj.get('anchors') if str(v).strip()]
                if len(anchors) >= 2:
                    low = low or anchors[0]
                    high = high or anchors[-1]

        if low and high and low.lower() != high.lower():
            self._log('suggest_continuous_anchors success low="%s" high="%s"', low, high)
            return {
                'low': low,
                'high': high,
                'provider': provider or 'unknown',
            }

        self._log('suggest_continuous_anchors failed: %s', self._last_error())
        return {
            'low': '',
            'high': '',
            'provider': provider or self.provider or 'unknown',
            'error': self._last_error(),
        }

    def suggest_values(
        self,
        attribute: str,
        n_values: int = DEFAULT_VALUE_COUNT,
        provided_values: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Backward-compatible wrapper; maps to ordinal category suggestion.
        out = self.suggest_categories(
            attribute=attribute,
            attribute_type='ordinal',
            n_values=n_values,
            provided_categories=provided_values,
        )
        return {
            'values': out.get('categories') or [],
            'provider': out.get('provider') or 'unknown',
            **({'error': out.get('error')} if out.get('error') else {}),
        }
