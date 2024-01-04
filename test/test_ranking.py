import unittest

from poprank import Rank


class TestRank(unittest.TestCase):

    def test_composition_with_identity(self):
        """
            Test the composition with the permutation
            group identity.
        """
        identity = Rank([0, 1, 2, 3, 4])
        rank = Rank([1, 4, 3, 2, 0])

        self.assertEqual(identity * rank, rank)
        self.assertEqual(rank * identity, rank)

    def test_inverse(self):
        rank = Rank([1, 4, 3, 2, 0])
        true_inverse = Rank([4, 0, 3, 2, 1])

        self.assertEqual(rank ** -1, true_inverse)
        self.assertEqual(true_inverse ** -1, rank)
        self.assertEqual((rank ** -1) ** -1, rank)



class TestRankIntegration(unittest.TestCase):
    """
        Verify the intergration of ratings, ranks, and metrics.
    """
    pass
