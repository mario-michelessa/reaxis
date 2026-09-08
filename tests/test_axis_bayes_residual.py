from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from backend.axis_bayes import (
    AxisBayesEngine,
    CollectionCache,
    _normalize_rows,
    _normalize_vec,
    _rank_percentile_01,
)


class StubResidualEngine(AxisBayesEngine):
    def __init__(self, collections: dict[str, CollectionCache], prompt_vec: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._collections_by_root = {str(Path(key).resolve()): value for key, value in collections.items()}
        self._stub_prompt = _normalize_vec(prompt_vec)

    def _get_collection(self, dataset_root: str, collection_id: str) -> CollectionCache:
        key = str(Path(dataset_root).resolve())
        return self._collections_by_root[key]

    def _embed_prompt_ensemble(
        self,
        q: str,
        semantic_method: str,
        norm: bool | None = None,
        dataset_name: str = '',
    ):
        return (
            self._stub_prompt.copy(),
            [f'high {q}'],
            [f'low {q}'],
            {'source': 'test', 'provider': 'test'},
        )


def _fused_collection(
    dataset_root: str,
    clip_raw: np.ndarray,
    dino_raw: np.ndarray,
    *,
    clip_weight: float = 0.7,
    dino_weight: float = 0.3,
) -> CollectionCache:
    clip = _normalize_rows(np.asarray(clip_raw, dtype=np.float32))
    dino = _normalize_rows(np.asarray(dino_raw, dtype=np.float32))
    total = float(max(1e-6, clip_weight) + max(1e-6, dino_weight))
    clip_scale = float(np.sqrt(max(1e-6, clip_weight) / total))
    dino_scale = float(np.sqrt(max(1e-6, dino_weight) / total))
    fused = np.concatenate([clip_scale * clip, dino_scale * dino], axis=1).astype(np.float32)
    ids = [f'img-{idx:03d}' for idx in range(fused.shape[0])]
    return CollectionCache(
        dataset_root=str(Path(dataset_root).resolve()),
        collection_id=Path(dataset_root).name,
        ids=ids,
        embeddings=fused,
        embedding_norms=np.linalg.norm(fused, axis=1).astype(np.float32),
        id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
        feature_space='clip_dino',
        semantic_method='clip',
        norm=True,
        clip_dim=int(clip.shape[1]),
        dino_dim=int(dino.shape[1]),
        clip_scale=clip_scale,
        dino_scale=dino_scale,
    )


class AxisBayesResidualTests(unittest.TestCase):
    def test_residual_without_labels_matches_clip_prior(self) -> None:
        clip = np.asarray(
            [
                [1.0, 0.0],
                [0.6, 0.4],
                [0.1, 0.9],
                [-0.3, 0.95],
            ],
            dtype=np.float32,
        )
        dino = np.asarray(
            [
                [0.9, 0.1],
                [0.5, 0.5],
                [0.2, 0.8],
                [-0.1, 1.0],
            ],
            dtype=np.float32,
        )
        coll = _fused_collection('/tmp/residual-prior', clip, dino)
        prompt = np.asarray([1.0, 0.0], dtype=np.float32)
        engine = StubResidualEngine(
            {'/tmp/residual-prior': coll},
            prompt,
            model_type='residual',
            mode='gaussian',
            feature_space='clip_dino',
            clip_weight=0.7,
            dino_weight=0.3,
        )
        created = engine.create_axis(coll.collection_id, '/tmp/residual-prior', 'smiling', model_type='residual')
        expected = _rank_percentile_01(_normalize_rows(clip) @ _normalize_vec(prompt))
        np.testing.assert_allclose(np.asarray(created['projection_values'], dtype=np.float32), expected, atol=1e-6)
        self.assertEqual(created['model_type'], 'residual')

    def test_residual_feedback_moves_labeled_example_and_reduces_std(self) -> None:
        clip = np.asarray(
            [
                [-1.0, -0.4],
                [-0.8, -0.2],
                [-0.2, 0.6],
                [0.0, 1.0],
                [0.2, 0.9],
                [0.8, 0.1],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        dino = np.asarray(
            [
                [-0.8, -0.1],
                [-0.7, 0.0],
                [-0.1, 0.9],
                [0.1, 1.0],
                [0.2, 0.8],
                [0.7, 0.2],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        )
        coll = _fused_collection('/tmp/residual-feedback', clip, dino)
        prompt = np.asarray([1.0, 0.0], dtype=np.float32)
        engine = StubResidualEngine(
            {'/tmp/residual-feedback': coll},
            prompt,
            model_type='residual',
            mode='gaussian',
            feature_space='clip_dino',
            clip_weight=0.7,
            dino_weight=0.3,
            residual_lambda=0.15,
            residual_sigma_y=0.05,
        )
        created = engine.create_axis(coll.collection_id, '/tmp/residual-feedback', 'tumor', model_type='residual')
        moved_idx = 3
        before_score = float(created['projection_values'][moved_idx])
        before_std = float(created['std'][moved_idx])
        moved = engine.move_axis(created['axis_id'], coll.ids[moved_idx], 100.0)
        after_score = float(moved['projection_values'][moved_idx])
        after_std = float(moved['std'][moved_idx])
        self.assertGreater(after_score, before_score + 0.15)
        self.assertLess(after_std, before_std)

    def test_residual_portable_projection_recomputes_target_dataset_scores(self) -> None:
        source = _fused_collection(
            '/tmp/residual-source',
            np.asarray([[1.0, 0.0], [0.6, 0.5], [-0.2, 1.0], [-0.8, 0.2]], dtype=np.float32),
            np.asarray([[0.9, 0.1], [0.5, 0.6], [0.0, 1.0], [-0.7, 0.3]], dtype=np.float32),
        )
        target_a = _fused_collection(
            '/tmp/residual-target-a',
            np.asarray([[0.9, 0.1], [0.1, 0.9], [-0.8, 0.2], [-0.2, -0.9]], dtype=np.float32),
            np.asarray([[0.8, 0.2], [0.2, 0.8], [-0.7, 0.1], [-0.1, -1.0]], dtype=np.float32),
        )
        target_b = _fused_collection(
            '/tmp/residual-target-b',
            np.asarray([[-0.9, 0.2], [-0.4, 0.8], [0.5, 0.6], [1.0, 0.0]], dtype=np.float32),
            np.asarray([[-0.8, 0.1], [-0.2, 0.9], [0.4, 0.7], [0.9, 0.1]], dtype=np.float32),
        )
        prompt = np.asarray([1.0, 0.0], dtype=np.float32)
        engine = StubResidualEngine(
            {
                '/tmp/residual-source': source,
                '/tmp/residual-target-a': target_a,
                '/tmp/residual-target-b': target_b,
            },
            prompt,
            model_type='residual',
            mode='gaussian',
            feature_space='clip_dino',
            clip_weight=0.7,
            dino_weight=0.3,
        )
        created = engine.create_axis(source.collection_id, '/tmp/residual-source', 'pneumonia', model_type='residual')
        engine.move_axis(created['axis_id'], source.ids[2], 100.0)
        portable = engine.serialize_portable_axis(created['axis_id'])

        blob_a = engine.project_serialized_axis(portable, target_a.collection_id, '/tmp/residual-target-a', 'Pneumonia')
        state_a = engine.deserialize_axis(blob_a)
        payload_a = engine._state_payload(state_a)

        blob_b = engine.project_serialized_axis(portable, target_b.collection_id, '/tmp/residual-target-b', 'Pneumonia')
        state_b = engine.deserialize_axis(blob_b)
        payload_b = engine._state_payload(state_b)

        self.assertEqual(payload_a['ids'], target_a.ids)
        self.assertEqual(payload_b['ids'], target_b.ids)
        self.assertEqual(len(payload_a['projection_values']), len(target_a.ids))
        self.assertEqual(len(payload_b['projection_values']), len(target_b.ids))
        self.assertFalse(
            np.allclose(
                np.asarray(payload_a['projection_values'], dtype=np.float32),
                np.asarray(payload_b['projection_values'], dtype=np.float32),
                atol=1e-6,
            )
        )


if __name__ == '__main__':
    unittest.main()
