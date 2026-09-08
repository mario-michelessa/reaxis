from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend import modeling_evaluation as me


class ModelingEvaluationConfigTests(unittest.TestCase):
    def _assert_close(self, left: float, right: float, *, msg: str = '') -> None:
        self.assertTrue(math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=2e-6), msg or f'{left} != {right}')

    def test_gaussian_method_uses_gaussian_sweep_winner(self) -> None:
        spec = me.METHOD_REGISTRY['request_bayes_linear_gaussian']
        engine = me.EvaluationContext([]).get_engine(spec)
        expected = me.GAUSSIAN_SWEEP_BEST

        self.assertEqual(spec.feature_space, expected['feature_space'])
        self._assert_close(spec.clip_weight, expected['clip_weight'])
        self._assert_close(spec.dino_weight, expected['dino_weight'])
        self._assert_close(spec.params['alpha'], expected['alpha'])
        self._assert_close(spec.params['dino_alpha'], expected['dino_alpha'])
        self._assert_close(spec.params['bias_alpha'], expected['bias_alpha'])
        self._assert_close(spec.params['sigma2'], expected['sigma2'])
        self._assert_close(engine.clip_weight, expected['clip_weight'])
        self._assert_close(engine.dino_weight, expected['dino_weight'])
        self._assert_close(engine.alpha, expected['alpha'])
        self._assert_close(engine.dino_alpha, expected['dino_alpha'])
        self._assert_close(engine.sigma2, expected['sigma2'])

    def test_rank_method_reuses_tuned_linear_prior_settings(self) -> None:
        spec = me.METHOD_REGISTRY['request_bayes_linear_rank']
        engine = me.EvaluationContext([]).get_engine(spec)
        expected = me.RANK_SWEEP_BEST

        self.assertEqual(spec.feature_space, expected['feature_space'])
        self._assert_close(spec.clip_weight, expected['clip_weight'])
        self._assert_close(spec.dino_weight, expected['dino_weight'])
        self._assert_close(spec.params['alpha'], expected['alpha'])
        self._assert_close(spec.params['dino_alpha'], expected['dino_alpha'])
        self._assert_close(spec.params['bias_alpha'], expected['bias_alpha'])
        self._assert_close(spec.params['sigma2'], expected['sigma2'])
        self._assert_close(spec.params['rank_eta'], expected['rank_eta'])
        self.assertEqual(spec.params['rank_anchor_k'], expected['rank_anchor_k'])
        self._assert_close(spec.params['rank_anchor_delta'], expected['rank_anchor_delta'])
        self.assertEqual(spec.params['rank_max_pairs'], expected['rank_max_pairs'])
        self._assert_close(engine.clip_weight, expected['clip_weight'])
        self._assert_close(engine.dino_weight, expected['dino_weight'])
        self._assert_close(engine.alpha, expected['alpha'])
        self._assert_close(engine.dino_alpha, expected['dino_alpha'])
        self._assert_close(engine.bias_alpha, expected['bias_alpha'])
        self._assert_close(engine.sigma2, expected['sigma2'])
        self._assert_close(engine.rank_eta, expected['rank_eta'])
        self.assertEqual(engine.rank_anchor_k, expected['rank_anchor_k'])
        self._assert_close(engine.rank_anchor_delta, expected['rank_anchor_delta'])
        self.assertEqual(engine.rank_max_pairs, expected['rank_max_pairs'])

    def test_graph_method_uses_graph_sweep_winner(self) -> None:
        spec = me.METHOD_REGISTRY['request_graph']
        engine = me.EvaluationContext([]).get_engine(spec)
        expected = me.GRAPH_SWEEP_BEST

        self.assertEqual(spec.feature_space, expected['feature_space'])
        self._assert_close(spec.clip_weight, expected['clip_weight'])
        self._assert_close(spec.dino_weight, expected['dino_weight'])
        self._assert_close(spec.params['sigma2'], expected['sigma2'])
        self.assertEqual(spec.params['graph_knn_k'], expected['graph_knn_k'])
        self._assert_close(spec.params['graph_lambda_smooth'], expected['graph_lambda_smooth'])
        self._assert_close(spec.params['graph_lambda_prior'], expected['graph_lambda_prior'])
        self._assert_close(spec.params['graph_jitter'], expected['graph_jitter'])
        self._assert_close(engine.clip_weight, expected['clip_weight'])
        self._assert_close(engine.dino_weight, expected['dino_weight'])
        self._assert_close(engine.sigma2, expected['sigma2'])
        self.assertEqual(engine.graph_knn_k, expected['graph_knn_k'])
        self._assert_close(engine.graph_lambda_smooth, expected['graph_lambda_smooth'])
        self._assert_close(engine.graph_lambda_prior, expected['graph_lambda_prior'])

    def test_residual_method_uses_explicit_defaults(self) -> None:
        spec = me.METHOD_REGISTRY['residual']
        engine = me.EvaluationContext([]).get_engine(spec)
        expected = me.RESIDUAL_DEFAULTS

        self.assertEqual(spec.feature_space, expected['feature_space'])
        self._assert_close(spec.clip_weight, expected['clip_weight'])
        self._assert_close(spec.dino_weight, expected['dino_weight'])
        self._assert_close(spec.params['residual_alpha'], expected['residual_alpha'])
        self._assert_close(spec.params['residual_beta'], expected['residual_beta'])
        self._assert_close(spec.params['residual_lambda'], expected['residual_lambda'])
        self._assert_close(spec.params['residual_sigma_y'], expected['residual_sigma_y'])
        self._assert_close(spec.params['residual_lengthscale_multiplier'], expected['residual_lengthscale_multiplier'])
        self._assert_close(spec.params['residual_jitter'], expected['residual_jitter'])
        self._assert_close(engine.residual_alpha, expected['residual_alpha'])
        self._assert_close(engine.residual_beta, expected['residual_beta'])
        self._assert_close(engine.residual_lambda, expected['residual_lambda'])
        self._assert_close(engine.residual_sigma_y, expected['residual_sigma_y'])
        self._assert_close(engine.residual_lengthscale_multiplier, expected['residual_lengthscale_multiplier'])
        self._assert_close(engine.residual_jitter, expected['residual_jitter'])

    def test_piecewise_method_uses_piecewise_sweep_winner(self) -> None:
        spec = me.METHOD_REGISTRY['request_piecewise']
        engine = me.EvaluationContext([]).get_engine(spec)
        expected = me.PIECEWISE_SWEEP_BEST

        self.assertEqual(spec.feature_space, expected['feature_space'])
        self._assert_close(spec.clip_weight, expected['clip_weight'])
        self._assert_close(spec.dino_weight, expected['dino_weight'])
        self.assertEqual(spec.params['piecewise_num_experts'], expected['piecewise_num_experts'])
        self.assertEqual(spec.params['piecewise_use_gating'], expected['piecewise_use_gating'])
        self.assertEqual(spec.params['piecewise_aggregator'], expected['piecewise_aggregator'])
        self._assert_close(spec.params['piecewise_clip_scale'], expected['piecewise_clip_scale'])
        self._assert_close(spec.params['piecewise_dino_scale'], expected['piecewise_dino_scale'])
        self._assert_close(spec.params['pairwise_from_scalar_margin'], expected['pairwise_from_scalar_margin'])
        self._assert_close(spec.params['piecewise_prior_strength'], expected['piecewise_prior_strength'])
        self._assert_close(spec.params['piecewise_expert_diversity_strength'], expected['piecewise_expert_diversity_strength'])
        self._assert_close(spec.params['piecewise_l2_reg'], expected['piecewise_l2_reg'])
        self._assert_close(spec.params['piecewise_learning_rate'], expected['piecewise_learning_rate'])
        self.assertEqual(spec.params['piecewise_max_refine_steps'], expected['piecewise_max_refine_steps'])
        self._assert_close(engine.clip_weight, expected['clip_weight'])
        self._assert_close(engine.dino_weight, expected['dino_weight'])
        self.assertEqual(engine.piecewise_num_experts, expected['piecewise_num_experts'])
        self.assertEqual(engine.piecewise_use_gating, expected['piecewise_use_gating'])
        self.assertEqual(engine.piecewise_aggregator, expected['piecewise_aggregator'])
        self._assert_close(engine.piecewise_clip_scale, expected['piecewise_clip_scale'])
        self._assert_close(engine.piecewise_dino_scale, expected['piecewise_dino_scale'])

    def test_compute_metrics_includes_ranking_coherence_and_binary_auroc(self) -> None:
        rng = np.random.default_rng(0)
        pred = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        ref = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        metrics = me.compute_metrics(pred, ref, rng=rng)

        self.assertAlmostEqual(metrics['auroc_binary'], 1.0, places=7)
        self.assertAlmostEqual(metrics['pairwise_acc'], 1.0, places=7)
        self.assertAlmostEqual(metrics['kendall_tau'], 1.0, places=7)

    def test_binary_prefix_and_value_task_builders(self) -> None:
        dataset = me.LoadedDataset(
            name='toy',
            dataset_root=Path('/tmp/toy'),
            ids=['a', 'b', 'c', 'd'],
            paths=['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
            id_to_index={'a': 0, 'b': 1, 'c': 2, 'd': 3},
            metadata=pd.DataFrame(
                {
                    'image': ['a', 'b', 'c', 'd'],
                    'attr_Smiling': [1, -1, 1, -1],
                    'attr_Bald': [-1, -1, 1, 1],
                    'label': ['yes', 'no', 'yes', 'no'],
                }
            ),
            image_col='image',
            X_clip=np.zeros((4, 2), dtype=np.float32),
            X_dino=np.zeros((4, 2), dtype=np.float32),
            fused_default=np.zeros((4, 4), dtype=np.float32),
        )

        prefix_tasks = me.build_metadata_binary_prefix_tasks(
            dataset,
            me.ConceptSpec(kind='metadata_binary_prefix', name='face_attribute', field_prefix='attr_', min_count=1),
        )
        value_tasks = me.build_metadata_binary_value_tasks(
            dataset,
            me.ConceptSpec(kind='metadata_binary_value', name='tumor', field='label', positive_values=('yes',), min_count=1),
        )

        self.assertEqual(sorted(task.concept_name for task in prefix_tasks), ['Bald', 'Smiling'])
        self.assertEqual(len(value_tasks), 1)
        np.testing.assert_array_equal(value_tasks[0].reference_scores01, np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
