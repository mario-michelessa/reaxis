from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from backend.axis_bayes import (
    AxisBayesEngine,
    CollectionCache,
    _normalize_rows,
    _normalize_vec,
    _rank_percentile_01,
)


def _percentile_targets(values: np.ndarray) -> np.ndarray:
    return _rank_percentile_01(np.asarray(values, dtype=np.float32)) * 100.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if np.std(aa) <= 1e-8 or np.std(bb) <= 1e-8:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


class StubAxisBayesEngine(AxisBayesEngine):
    def __init__(self, collection: CollectionCache, prompt_vec: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._stub_collection = collection
        self._stub_prompt = _normalize_vec(prompt_vec)

    def _get_collection(self, dataset_root: str, collection_id: str) -> CollectionCache:
        return self._stub_collection

    def _embed_prompt_ensemble(self, q: str):
        return (
            self._stub_prompt.copy(),
            [f'high {q}'],
            [f'low {q}'],
            {'source': 'test', 'provider': 'test'},
        )

    def _embed_prompt_lists(self, pos_prompts, neg_prompts, *, source: str, provider: str):
        pos_list = self._clean_prompt_list(pos_prompts)
        neg_list = self._clean_prompt_list(neg_prompts)
        axis_x = sum('x' in prompt.lower() or 'horizontal' in prompt.lower() for prompt in pos_list)
        axis_y = sum('y' in prompt.lower() or 'vertical' in prompt.lower() for prompt in pos_list)
        neg_x = sum('x' in prompt.lower() or 'horizontal' in prompt.lower() for prompt in neg_list)
        neg_y = sum('y' in prompt.lower() or 'vertical' in prompt.lower() for prompt in neg_list)
        vec = np.asarray([
            float(axis_x - neg_x),
            float(axis_y - neg_y),
            0.0,
        ], dtype=np.float32)
        if float(np.linalg.norm(vec)) <= 1e-8:
            vec = self._stub_prompt.copy()
        else:
            vec = _normalize_vec(vec)
        return vec, pos_list, neg_list, {'source': source, 'provider': provider}


class AxisBayesPiecewiseTests(unittest.TestCase):
    def _collection(self, embeddings: np.ndarray) -> CollectionCache:
        X = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
        ids = [f'img-{idx:03d}' for idx in range(X.shape[0])]
        return CollectionCache(
            dataset_root='/tmp/test-dataset',
            collection_id='test-dataset',
            ids=ids,
            embeddings=X.astype(np.float32),
            id_to_index={image_id: idx for idx, image_id in enumerate(ids)},
            feature_space='clip',
            clip_dim=int(X.shape[1]),
            dino_dim=0,
            clip_scale=1.0,
            dino_scale=0.0,
        )

    def _make_linear_problem(self, n: int = 72):
        rng = np.random.default_rng(42)
        xy = rng.normal(size=(n, 2)).astype(np.float32)
        emb = np.concatenate([xy, np.full((n, 1), 0.25, dtype=np.float32)], axis=1)
        true_score = xy[:, 0].astype(np.float32)
        prompt = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return self._collection(emb), prompt, true_score

    def _make_piecewise_problem(self, n: int = 96):
        rng = np.random.default_rng(7)
        xy = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
        emb = np.concatenate([xy, np.full((n, 1), 0.2, dtype=np.float32)], axis=1)
        true_score = np.maximum(xy[:, 0], xy[:, 1]).astype(np.float32)
        prompt = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return self._collection(emb), prompt, true_score

    def _label_indices(self, true_score: np.ndarray, count: int = 8) -> list[int]:
        order = np.argsort(true_score)
        picks = np.linspace(0, len(order) - 1, count, dtype=int)
        return [int(order[idx]) for idx in picks.tolist()]

    def _apply_labels(self, engine: AxisBayesEngine, axis_id: str, ids: list[str], target_pct: np.ndarray):
        result = None
        for image_id, pct in zip(ids, target_pct):
            result = engine.move_axis(axis_id=axis_id, image_id=image_id, new_score_0_100=float(pct))
        return result

    def test_legacy_default_matches_explicit_bayes_linear(self):
        coll, prompt, true_score = self._make_linear_problem()
        default_engine = StubAxisBayesEngine(
            coll,
            prompt,
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        explicit_engine = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='bayes_linear',
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        default_axis = default_engine.create_axis('test-dataset', '/tmp/test-dataset', 'sharpness')
        explicit_axis = explicit_engine.create_axis('test-dataset', '/tmp/test-dataset', 'sharpness', model_type='bayes_linear')
        labels = self._label_indices(true_score, count=4)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])
        default_payload = self._apply_labels(default_engine, default_axis['axis_id'], ids, pct)
        explicit_payload = self._apply_labels(explicit_engine, explicit_axis['axis_id'], ids, pct)
        self.assertEqual(default_payload['model_type'], 'bayes_linear')
        np.testing.assert_allclose(default_payload['projection_values'], explicit_payload['projection_values'], atol=1e-5)
        np.testing.assert_allclose(default_payload['scores'], explicit_payload['scores'], atol=1e-5)

    def test_piecewise_two_experts_beats_single_expert_on_nonlinear_ranking(self):
        coll, prompt, true_score = self._make_piecewise_problem()
        one_expert_engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=1,
            piecewise_aggregator='max',
            piecewise_max_refine_steps=24,
            rank_max_pairs=256,
        )
        two_expert_engine = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_use_gating=False,
            piecewise_aggregator='max',
            piecewise_max_refine_steps=24,
            rank_max_pairs=256,
        )
        one_expert_axis = one_expert_engine.create_axis('test-dataset', '/tmp/test-dataset', 'vshape', model_type='piecewise_linear')
        two_expert_axis = two_expert_engine.create_axis('test-dataset', '/tmp/test-dataset', 'vshape', model_type='piecewise_linear')
        labels = self._label_indices(true_score, count=8)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])
        one_expert_payload = self._apply_labels(one_expert_engine, one_expert_axis['axis_id'], ids, pct)
        two_expert_payload = self._apply_labels(two_expert_engine, two_expert_axis['axis_id'], ids, pct)
        one_expert_corr = _corr(_rank_percentile_01(one_expert_payload['projection_values']), _rank_percentile_01(true_score))
        two_expert_corr = _corr(_rank_percentile_01(two_expert_payload['projection_values']), _rank_percentile_01(true_score))
        self.assertGreater(two_expert_corr, one_expert_corr + 0.2)
        self.assertEqual(two_expert_payload['model_type'], 'piecewise_linear')

    def test_piecewise_uncertainty_on_labeled_examples_decreases(self):
        coll, prompt, true_score = self._make_piecewise_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=18,
        )
        created = engine.create_axis('test-dataset', '/tmp/test-dataset', 'curvature', model_type='piecewise_linear')
        labels = self._label_indices(true_score, count=6)
        before = np.asarray(created['std'], dtype=np.float32)[labels]
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])
        moved = self._apply_labels(engine, created['axis_id'], ids, pct)
        after = np.asarray(moved['std'], dtype=np.float32)[labels]
        self.assertLess(float(np.mean(after)), float(np.mean(before)))

    def test_rankings_update_after_feedback(self):
        coll, prompt, true_score = self._make_linear_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=15,
        )
        created = engine.create_axis('test-dataset', '/tmp/test-dataset', 'contrast', model_type='piecewise_linear')
        scores_before = np.asarray(created['scores'], dtype=np.float32)
        low_idx = int(np.argmin(scores_before))
        moved = engine.move_axis(created['axis_id'], coll.ids[low_idx], 100.0)
        scores_after = np.asarray(moved['scores'], dtype=np.float32)
        self.assertGreater(scores_after[low_idx], scores_before[low_idx] + 10.0)

    def test_undefined_examples_are_excluded_from_feedback_and_payload(self):
        coll, prompt, true_score = self._make_linear_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='bayes_linear',
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        created = engine.create_axis('test-dataset', '/tmp/test-dataset', 'ambiguity')
        label_idx = self._label_indices(true_score, count=1)[0]
        image_id = coll.ids[label_idx]
        moved = engine.move_axis(created['axis_id'], image_id, 100.0)
        self.assertEqual(moved['move_count'], 1)
        undefined = engine.move_axis(created['axis_id'], image_id, 0.0, move_type='undefined')
        self.assertEqual(undefined['move_count'], 0)
        self.assertIn(image_id, undefined['undefined_ids'])
        exemplar_ids = {
            exemplar['id']
            for decile in undefined['decile_exemplars']
            for exemplar in (decile.get('exemplars') or [])
        }
        self.assertNotIn(image_id, exemplar_ids)

        blob = engine.serialize_axis(created['axis_id'])
        restored = StubAxisBayesEngine(
            self._collection(coll.embeddings),
            prompt,
            model_type='bayes_linear',
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        state = restored.deserialize_axis(blob)
        round_trip = restored._state_payload(state)
        self.assertIn(image_id, round_trip['undefined_ids'])

    def test_piecewise_undefined_feedback_does_not_crash_anchor_pair_building(self):
        coll, prompt, true_score = self._make_piecewise_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=12,
        )
        created = engine.create_axis('test-dataset', '/tmp/test-dataset', 'piecewise-undefined', model_type='piecewise_linear')
        labels = self._label_indices(true_score, count=4)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])
        refined = self._apply_labels(engine, created['axis_id'], ids, pct)
        undefined_payload = engine.move_axis(refined['axis_id'], ids[0], 0.0, move_type='undefined')
        self.assertIn(ids[0], undefined_payload['undefined_ids'])
        self.assertEqual(undefined_payload['model_type'], 'piecewise_linear')
        self.assertEqual(undefined_payload['move_count'], len(ids) - 1)

    def test_prompt_update_rebuilds_linear_prior_and_preserves_moves(self):
        coll, prompt, true_score = self._make_linear_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='bayes_linear',
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        created = engine.create_axis('test-dataset', '/tmp/test-dataset', 'direction')
        moved = engine.move_axis(created['axis_id'], coll.ids[0], 100.0)
        updated = engine.update_axis_prompts(
            created['axis_id'],
            pos_prompts=['strong vertical'],
            neg_prompts=['strong horizontal'],
        )
        self.assertEqual(updated['move_count'], 1)
        self.assertEqual(updated['w0_summary']['pos_prompts'], ['strong vertical'])
        self.assertEqual(updated['w0_summary']['neg_prompts'], ['strong horizontal'])
        self.assertEqual(updated['w0_summary']['prompt_source'], 'manual_edit')
        self.assertNotEqual(
            np.argmax(np.abs(np.asarray(moved['projection_values'], dtype=np.float32))),
            -1,
        )
        diff = np.max(
            np.abs(
                np.asarray(updated['projection_values'], dtype=np.float32)
                - np.asarray(moved['projection_values'], dtype=np.float32)
            )
        )
        self.assertGreater(float(diff), 1e-4)

    def test_prompt_update_rebuilds_piecewise_prior_and_preserves_feedback(self):
        coll, prompt, true_score = self._make_piecewise_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=12,
        )
        created = engine.create_axis('test-dataset', '/tmp/test-dataset', 'direction', model_type='piecewise_linear')
        labels = self._label_indices(true_score, count=4)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])
        refined = self._apply_labels(engine, created['axis_id'], ids, pct)
        updated = engine.update_axis_prompts(
            created['axis_id'],
            pos_prompts=['strong vertical'],
            neg_prompts=['strong horizontal'],
        )
        self.assertEqual(updated['move_count'], len(ids))
        self.assertEqual(updated['model_type'], 'piecewise_linear')
        self.assertEqual(updated['w0_summary']['pos_prompts'], ['strong vertical'])
        diff = np.max(
            np.abs(
                np.asarray(updated['projection_values'], dtype=np.float32)
                - np.asarray(refined['projection_values'], dtype=np.float32)
            )
        )
        self.assertGreater(float(diff), 1e-4)

    def test_serialization_round_trip_for_legacy_and_piecewise(self):
        coll, prompt, true_score = self._make_linear_problem()
        labels = self._label_indices(true_score, count=4)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])

        legacy_engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='bayes_linear',
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        legacy_created = legacy_engine.create_axis('test-dataset', '/tmp/test-dataset', 'legacy')
        legacy_payload = self._apply_labels(legacy_engine, legacy_created['axis_id'], ids, pct)
        legacy_blob = legacy_engine.serialize_axis(legacy_created['axis_id'])
        legacy_restored = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='bayes_linear',
            mode='gaussian',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
        )
        legacy_state = legacy_restored.deserialize_axis(legacy_blob)
        legacy_round_trip = legacy_restored._state_payload(legacy_state)
        np.testing.assert_allclose(legacy_payload['projection_values'], legacy_round_trip['projection_values'], atol=1e-5)

        piecewise_engine = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=18,
        )
        piecewise_created = piecewise_engine.create_axis('test-dataset', '/tmp/test-dataset', 'piecewise', model_type='piecewise_linear')
        piecewise_payload = self._apply_labels(piecewise_engine, piecewise_created['axis_id'], ids, pct)
        piecewise_blob = piecewise_engine.serialize_axis(piecewise_created['axis_id'])
        piecewise_restored = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
        )
        piecewise_state = piecewise_restored.deserialize_axis(piecewise_blob)
        piecewise_round_trip = piecewise_restored._state_payload(piecewise_state)
        np.testing.assert_allclose(piecewise_payload['projection_values'], piecewise_round_trip['projection_values'], atol=1e-5)

    def test_project_serialized_piecewise_axis_preserves_refined_state(self):
        coll, prompt, true_score = self._make_piecewise_problem()
        labels = self._label_indices(true_score, count=8)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])

        save_engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=18,
        )
        created = save_engine.create_axis('test-dataset', '/tmp/test-dataset', 'piecewise-save', model_type='piecewise_linear')
        refined_payload = self._apply_labels(save_engine, created['axis_id'], ids, pct)
        artifact = save_engine.serialize_portable_axis(created['axis_id'])
        self.assertEqual(artifact.get('artifact_scope'), 'portable')
        self.assertNotIn('image_embeddings', artifact)
        self.assertNotIn('ids', artifact)
        self.assertNotIn('move_targets', artifact)
        self.assertNotIn('z0_all', artifact)
        self.assertNotIn('z0_sorted', artifact)

        prior_engine = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=18,
        )
        prior_payload = prior_engine.create_axis('test-dataset', '/tmp/test-dataset', 'piecewise-save', model_type='piecewise_linear')

        project_engine = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
            piecewise_max_refine_steps=18,
        )
        projected_blob = project_engine.project_serialized_axis(
            artifact,
            collection_id='test-dataset',
            dataset_root='/tmp/test-dataset',
            axis_name='piecewise-save',
        )
        projected_state = project_engine._axes[projected_blob['axis_id']]
        projected_payload = project_engine._state_payload(projected_state)

        np.testing.assert_allclose(
            refined_payload['projection_values'],
            projected_payload['projection_values'],
            atol=1e-5,
        )
        prior_delta = np.max(
            np.abs(
                np.asarray(refined_payload['projection_values'], dtype=np.float32)
                - np.asarray(prior_payload['projection_values'], dtype=np.float32)
            )
        )
        self.assertGreater(float(prior_delta), 1e-4)

    def test_piecewise_k1_tracks_linear_ranker(self):
        coll, prompt, true_score = self._make_linear_problem()
        labels = self._label_indices(true_score, count=6)
        ids = [coll.ids[idx] for idx in labels]
        pct = _percentile_targets(true_score[labels])
        linear_engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='bayes_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            rank_max_pairs=128,
        )
        pw1_engine = StubAxisBayesEngine(
            replace(coll),
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=1,
            piecewise_aggregator='max',
            piecewise_max_refine_steps=18,
        )
        linear_created = linear_engine.create_axis('test-dataset', '/tmp/test-dataset', 'linear-k1')
        pw1_created = pw1_engine.create_axis('test-dataset', '/tmp/test-dataset', 'linear-k1', model_type='piecewise_linear')
        linear_payload = self._apply_labels(linear_engine, linear_created['axis_id'], ids, pct)
        pw1_payload = self._apply_labels(pw1_engine, pw1_created['axis_id'], ids, pct)
        similarity = _corr(_rank_percentile_01(linear_payload['projection_values']), _rank_percentile_01(pw1_payload['projection_values']))
        self.assertGreater(similarity, 0.9)

    def test_ui_facing_payload_shape_stays_intact(self):
        coll, prompt, _ = self._make_linear_problem()
        engine = StubAxisBayesEngine(
            coll,
            prompt,
            model_type='piecewise_linear',
            mode='rank',
            feature_space='clip',
            clip_weight=1.0,
            dino_weight=0.0,
            piecewise_num_experts=2,
        )
        payload = engine.create_axis('test-dataset', '/tmp/test-dataset', 'structure', model_type='piecewise_linear')
        required = {'axis_id', 'axis', 'ids', 'projection_values', 'scores', 'std', 'decile_exemplars', 'hotspots', 'model_type'}
        self.assertTrue(required.issubset(payload.keys()))
        self.assertEqual(payload['axis']['model_type'], 'piecewise_linear')
        self.assertEqual(len(payload['projection_values']), len(coll.ids))
        self.assertEqual(len(payload['scores']), len(coll.ids))
        self.assertEqual(len(payload['std']), len(coll.ids))


if __name__ == '__main__':
    unittest.main()
