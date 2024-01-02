import unittest
# internal
from poprank.metrics import (
    kendall, footrule, corr,
    hamming, max, lee, cayley, ulam
)

from fixtures.loader import load_fixture


class TestMetrics(unittest.TestCase):

    def test_kendall_tau_base(self):
        x = [1, 2, 3, 4, 5]

        tau = kendall(x, x)

        self.assertEqual(tau, 0)

    def test_kendall_tau_wikipedia(self):
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 1, 2, 5]

        tau = kendall(x, y)

        self.assertEqual(tau, 4)

    def test_kendall_tau_wikipedia_normalization(self):
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 1, 2, 5]

        tau = kendall(x, y, normalize=True)

        self.assertEqual(tau, 0.4)

    def test_kendall_tau_maximal(self):
        """
            Test that inverted rank maximizes Kendall tau.

            Example
            -----
        """
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]

        tau = kendall(x, y)

        self.assertEqual(tau, 10)

    def test_kendall_tau_maximal_normalized(self):
        """
            Test that inverted rank maximizes normalized Kendall tau.
        """
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]

        tau = kendall(x, y, normalize=True)

        self.assertEqual(tau, 1.0)

    def test_kendall_tau_non_identity(self):
        """
            Tests the example on Kumar & Vassilvitskii slides.

            Example
            ------------------
            Consider a set of four colors: {Blue, Green, Red, Yellow}.
            The input list [B, G, R, Y] is the identity rank or order.
                    I  R1  R2
                    ---------
                B   1   1   2
                G   2   3   1
                R   3   2   3
                Y   4   4   4
            Kendall tau computes the distance between rankings R1 and R2.
                K(R1, R2) = 2
        """
        x = [1, 3, 2, 4]
        y = [2, 1, 3, 4]

        tau = kendall(x, y)

        self.assertEqual(tau, 2)

    def test_kendall_ml_wiki(self):
        x = [2, 1, 3, 4, 5]
        y = [5, 3, 4, 1, 2]

        tau = kendall(x, y)

        self.assertEqual(tau, 7)

    def test_kendall_tau_simple(self):
        x = [3, 1, 2]
        y = [2, 1, 3]

        tau = kendall(x, y)

        self.assertEqual(tau, 1)

    def test_spearman_footrule_base(self):
        x = [1, 2, 3, 4, 5]

        f = footrule(x, x)

        self.assertEqual(f, 0)

    def test_hamming_base(self):
        x = [1, 2, 3, 4, 5]

        f = hamming(x, x)

        self.assertEqual(f, 0)

    def test_hamming_base_max(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]

        f = hamming(x, y)

        self.assertEqual(f, 4)

    def test_diaconis_multiple_to_identity(self):
        """
            Verify the implementation of the following metrics
              - cayley
              - kendall
              - hamming
              - footrule
              - corr
              - ulam
            against Table 1 pp. 113 on [1].

        [1] Diaconis, P. "Group Representations in Probability
        and Statistics". Institute of Mathematical Statistics, 1988.
        """

        table = load_fixture("diaconis.metrics")

        assert len(table) == 24

        identity = [1, 2, 3, 4]
        metrics = [
            kendall, footrule, corr, hamming,
            # cayley, ulam
        ]

        for entry in table:
            perm = entry['perm']
            for metric in metrics:
                value = metric(identity, perm)
                truth = entry['metrics'][metric.__name__]
                self.assertEqual(
                    value, truth,
                    f"For {metric.__name__}, value: {value}, truth: {truth}"
                )
