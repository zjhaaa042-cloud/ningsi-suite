"""Regression checks for nested grouped validation and missing-pair handling."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import attlapse_two_channel_eval as ev
from ningsi.models.logistic import roc_auc


class TwoChannelEvaluation(unittest.TestCase):
    def test_pair_objective_abstentions_contribute_only_chance(self):
        y = np.array([0, 0, 1, 1, 0, 1, 1])
        people = np.array(list("AAAABBC"))
        scores = np.array([.1, np.nan, .8, np.nan, .5, np.nan, 0.])
        self.assertEqual(ev.selection_score(y, scores, people), (.5625, 1))
        scores[0] = .8
        self.assertEqual(ev.selection_score(y, scores, people), (.5, 1))

    def test_fitting_never_uses_heldout_people_or_features(self):
        gen = np.random.default_rng(3)
        groups = np.repeat([f"P{i}" for i in range(10)], 4)
        x = gen.normal(size=(len(groups), 3)); y = np.tile([0, 1, 1, 0], 10)
        choices = [ev.Candidate("strict", "pre12_2", "bipolar6", fam, params[0])
                   for fam, params in ev.PARAMETERS.items()]
        train = np.arange(32)
        first = ev.select(train, groups, y, {choices[0].matrix_key: x}, choices, seed=37)
        y[32:] = 1 - y[32:]
        x[32:] += 1.e7
        second = ev.select(train, groups, y, {choices[0].matrix_key: x}, choices, seed=37)
        self.assertEqual(first, second)
        holdouts = first[2]
        self.assertEqual(sorted(p for fold in holdouts for p in fold), sorted(set(groups[train])))
        self.assertFalse(any(p in set(groups[32:]) for fold in holdouts for p in fold))
        for c in choices:
            self.assertIsNone(ev.fit(x[:1], y[:1], groups[:1], c))
            state = ev.fit(x[train], y[train], groups[train], c)
            values = ev.predict(state, np.array([[np.nan]*3, [1., 2., 3.]]))
            self.assertTrue(np.isnan(values[0]))
            self.assertTrue(0 < values[1] < 1)
            np.testing.assert_array_equal(ev.predict(json.loads(json.dumps(state)), x[:7]), ev.predict(state, x[:7]))

    def test_cluster_bootstrap_matches_expanded_pairs_with_ties_and_abstention(self):
        groups = np.array(list("AAABBBCCCC")); y = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([.5, .9, np.nan, .4, .7, .7, .9, .5, .1, np.nan])
        counts = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0], [0, 0, 3]])
        desc, boot = ev.describe(y, scores, groups, counts)
        for i, multiplicity in enumerate(counts):
            take = np.concatenate([np.tile(np.flatnonzero((groups == p) & np.isfinite(scores)), n)
                                   for p, n in zip(np.unique(groups), multiplicity)])
            self.assertAlmostEqual(boot["pooled_auc"][i], roc_auc(y[take], scores[take]))
        ref = (desc, boot)
        compared = ev.paired(y, scores, scores, groups, counts, ref, ref)
        self.assertEqual(compared["same_probes"], 8)
        self.assertEqual(compared["common"]["pooled_auc"]["paired_cluster_ci95"], [0., 0.])
        self.assertEqual(compared["native_coverage"]["adjusted_auc_delta"], 0.)


if __name__ == "__main__":
    unittest.main()
