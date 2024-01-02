import unittest
from random import choices
from string import ascii_letters
# internal
from popcore import Interaction, Team
from poprank.functional.trueskill import trueskill, TrueSkillRate

from fixtures.loader import load_fixture

PRECISION = 5


class TestTrueskillFunctional(unittest.TestCase):
    def test_winning_increases_rating(self) -> None:
        """Default single interaction win case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [1, 0])]
        ratings = [TrueSkillRate(25, 25/3), TrueSkillRate(25, 25/3)]
        expected_results = [
            TrueSkillRate(29.39583169299151, 7.17147580700922),
            TrueSkillRate(20.604168307008482, 7.17147580700922)
        ]
        results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
                for x in results],
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
                for x in expected_results])

    def test_draw(self) -> None:
        """Default single interaction draw case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [.5, .5])]
        ratings = [TrueSkillRate(25, 25/3), TrueSkillRate(25, 25/3)]
        expected_results = [
            TrueSkillRate(24.999999999999993, 6.457515683245051),
            TrueSkillRate(24.999999999999993, 6.457515683245051)
        ]
        results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
                for x in results],
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
                for x in expected_results])

    def test_losing_decreases_rating(self) -> None:
        """Default single interaction loss case"""
        players = ["a", "b"]
        interactions = [Interaction(["a", "b"], [0, 1])]
        ratings = [
            TrueSkillRate(25, 25/3), TrueSkillRate(25, 25/3)
        ]
        expected_results = [
            TrueSkillRate(20.604168307008482, 7.17147580700922),
            TrueSkillRate(29.39583169299151, 7.17147580700922)
        ]
        results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
             for x in results],
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
             for x in expected_results])

    def test_against_trueskill_library_implementation(self) -> None:
        "Complicated interaction involving many teams of different sizes"
        players = [
            Team(id="1", members=["a", "b"]),
            "c",
            Team(id="2", members=["d", "e", "f"]),
            Team(id="3", members=["g", "h"])
        ]
        interactions = [
            Interaction(
                players=["1", "c", "2", "3"],
                outcomes=[1, 2, 2, 3]
            )
        ]
        ratings = [
            [  # Team 1
                TrueSkillRate(25, 25/3), TrueSkillRate(25, 25/3)
            ],
            TrueSkillRate(25, 25/3),  # Player C
            [  # Team 2
                TrueSkillRate(29, 25/3),
                TrueSkillRate(25, 8),
                TrueSkillRate(20, 25/3)
            ],
            [  # Team 3
                TrueSkillRate(25, 25/3),
                TrueSkillRate(25, 25/3)
            ]
        ]
        expected_results = [
            [  # Team 1
                TrueSkillRate(17.98545418246194, 7.249488170861282),
                TrueSkillRate(17.98545418246194, 7.249488170861282)
            ],
            TrueSkillRate(38.188106500904695, 6.503173524922751),  # Player C
            [  # Team 2
                TrueSkillRate(20.166629601014503, 7.33719008859177),
                TrueSkillRate(16.859096593595705, 7.123373334507644),
                TrueSkillRate(11.166629601014504, 7.33719008859177)
            ],
            [
                TrueSkillRate(27.659809715618746, 7.5964444225283145),
                TrueSkillRate(27.659809715618746, 7.5964444225283145)
            ]
        ]
        results = trueskill(players, interactions, ratings)

        self.assertListEqual(
            # Rounding for floating point tolerance
            [[TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
              for x in y]
             if isinstance(y, list) else
             TrueSkillRate(round(y.mu, PRECISION), round(y.std, PRECISION))
             for y in results],
            [[TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
              for x in y]
             if isinstance(y, list) else
             TrueSkillRate(round(y.mu, PRECISION), round(y.std, PRECISION))
             for y in expected_results])

    def test_against_trueskill_library_on_full_tournament(self):
        data = load_fixture("synthetic.trueskill.tournament")

        interactions = [
            Interaction(
                players=[
                    Team(
                        id="".join(choices(ascii_letters, k=10)),
                        members=t) for t in interaction["players"]
                ],
                outcomes=interaction["outcomes"]
            ) for interaction in data["interactions"]
        ]

        ratings = [TrueSkillRate(25, 25/3) for x in data["players"]]

        results = trueskill(data["players"], interactions, ratings)

        expected_results = [
            TrueSkillRate(51.90633032568547, 5.152291913349492),
            TrueSkillRate(-0.14687999106540617, 0.9365996379522524),
            TrueSkillRate(0.4540650480905728, 0.9311947030252915),
            TrueSkillRate(0.13313566468497742, 0.9304144846514893),
            TrueSkillRate(-0.7736780130962938, 0.92086459345153),
            TrueSkillRate(-0.5018216589497285, 0.9485525720071926),
            TrueSkillRate(-0.35077741217085373, 0.9288505181979116),
            TrueSkillRate(0.28311953890041597, 0.9378992714141935),
            TrueSkillRate(1.0882886073107747, 0.916306838738594),
            TrueSkillRate(0.5443461596913228, 0.9357279979765744),
            TrueSkillRate(-0.6461492028820802, 0.9440604773084467),
            TrueSkillRate(0.7033417455527605, 0.9331052344504057),
            TrueSkillRate(0.6610986541549926, 0.9289346121974258),
            TrueSkillRate(-0.36342655962184184, 0.9265334151336572),
            TrueSkillRate(0.48616705037407654, 0.9210727959787768),
            TrueSkillRate(0.6345274056468461, 0.9272344877371177),
            TrueSkillRate(0.049530589203413064, 0.9351551557875168),
            TrueSkillRate(-0.15527817402809876, 0.9295633594749055),
            TrueSkillRate(0.9121976600444945, 0.9251203660803101),
            TrueSkillRate(-0.013125637276157424, 0.9357784587476502),
            TrueSkillRate(0.6659428284211067, 0.925521673005709),
            TrueSkillRate(-0.2824424589989811, 0.9290250730425014),
            TrueSkillRate(-0.4629376609652222, 0.9235047606217096),
            TrueSkillRate(-0.3923957883947142, 0.9255975163061033),
            TrueSkillRate(-0.5923678314894874, 0.9015455536433515),
            TrueSkillRate(0.25030297875112184, 0.916379283032207)
        ]

        self.assertListEqual(
            # Rounding for floating point tolerance
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
             for x in results],
            [TrueSkillRate(round(x.mu, PRECISION), round(x.std, PRECISION))
             for x in expected_results])
