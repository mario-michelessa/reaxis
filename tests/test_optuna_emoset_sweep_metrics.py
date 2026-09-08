from __future__ import annotations

import unittest

import numpy as np

from backend import optuna_emoset_sweep as sweep


class OptunaEmoSetSweepMetricTests(unittest.TestCase):
    def test_binary_auroc_handles_perfect_and_tied_rankings(self) -> None:
        perfect = sweep.binary_auroc(
            scores=np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
        tied = sweep.binary_auroc(
            scores=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(perfect, 1.0, places=7)
        self.assertAlmostEqual(tied, 0.5, places=7)

    def test_run_experiment_reports_macro_auroc(self) -> None:
        rng = np.random.default_rng(0)
        emotions = ['a', 'b', 'c']
        per_emotion = 12
        ids = []
        labels = []
        for emo in emotions:
            for idx in range(per_emotion):
                ids.append(f'{emo}_{idx}')
                labels.append(emo)
        labels_arr = np.asarray(labels)
        Y = np.stack([(labels_arr == emo).astype(np.float32) for emo in emotions], axis=1)
        clip = rng.normal(size=(len(ids), 8)).astype(np.float32)
        dino = rng.normal(size=(len(ids), 12)).astype(np.float32)
        clip /= np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8
        dino /= np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8
        data = sweep.LoadedData(
            ids=ids,
            emotions=emotions,
            labels=labels_arr,
            id_to_idx={image_id: i for i, image_id in enumerate(ids)},
            id_to_emotion={image_id: image_id.split('_', 1)[0] for image_id in ids},
            Y=Y,
            X_clip=clip,
            X_dino=dino,
        )
        w_clip_text = {}
        for emo in emotions:
            vec = rng.normal(size=(clip.shape[1],)).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            w_clip_text[emo] = vec

        res = sweep.run_experiment(
            data=data,
            w_clip_text=w_clip_text,
            move_catalog=sweep.build_move_catalog(data, 2, 123),
            mode='rank',
            feature_space='clip_dino',
            clip_weight=0.5,
            dino_weight=0.5,
            alpha=0.1,
            dino_alpha=0.1,
            bias_alpha=float('nan'),
            sigma2=float('nan'),
            graph_knn_k=16,
            graph_lambda_smooth=6.0,
            graph_lambda_prior=1.0,
            graph_jitter=1e-6,
            rank_eta=0.25,
            rank_anchor_k=6,
            rank_anchor_delta=0.12,
            rank_max_pairs=64,
        )

        self.assertIn('final_macro_auroc', res)
        self.assertIn('macro_auroc', res)
        self.assertIn('final_diag_sum', res)
        self.assertEqual(len(res['macro_auroc']), len(res['diag_sum']))
        self.assertTrue(0.0 <= float(res['final_macro_auroc']) <= 1.0)


if __name__ == '__main__':
    unittest.main()
