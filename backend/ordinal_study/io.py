from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CSVAppender:
    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open('r', newline='', encoding='utf-8') as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
            if list(header) != self.fieldnames:
                raise RuntimeError(f'CSV schema mismatch for {self.path}: {header} != {self.fieldnames}')
        else:
            with self.path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row: Mapping[str, object]) -> None:
        clean = {field: row.get(field, '') for field in self.fieldnames}
        with self.path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(clean)
            handle.flush()


def read_completed_metric_keys(path: Path, run_id: str) -> set[tuple[str, str, str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[tuple[str, str, str, int, int]] = set()
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'run_id', 'dataset', 'axis_field', 'method', 'replicate', 'budget'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f'Metrics CSV missing required columns: {path}')
        for row in reader:
            if str(row.get('run_id', '')) != str(run_id):
                continue
            completed.add((
                str(row['dataset']),
                str(row['axis_field']),
                str(row['method']),
                int(row['replicate']),
                int(row['budget']),
            ))
    return completed


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def append_many(appender: CSVAppender, rows: Iterable[Mapping[str, object]]) -> None:
    for row in rows:
        appender.append(row)
