import unittest
from popcore import Interaction
from popcore.population import Population
from poprank import EloRate
from poprank.functional import bayeselo
import axelrod as axl   # type: ignore


def score_to_chess(outcome):
    if outcome[0] > outcome[1]:
        return (1, 0)
    if outcome[0] < outcome[1]:
        return (0, 1)
    return (0.5, 0.5)


class TestAxelrodImplementation(unittest.TestCase):

    def test_axelrod_bayeselo(self):

        players = [axl.Cooperator(),
                   axl.Defector(),
                   axl.TitForTat(),
                   axl.Grudger(),
                   axl.Alternator(),
                   axl.Aggravater(),
                   axl.Adaptive(),
                   axl.AlternatorHunter(),
                   axl.ArrogantQLearner(),
                   axl.Bully()]

        pop = Population()

        branches = [pop.branch(str(p), auto_rename=True) for p in players]

        for p, b in zip(players, branches):
            pop.checkout(b)
            pop.commit(p)

        for gen in range(7):
            interactions = []
            elos = [EloRate(mu=0., std=0.) for _ in players]

            tournament = axl.Tournament(players)
            results = tournament.play(progress_bar=False)

            for i, p1 in enumerate(players):
                for j, p2 in enumerate(players[i+1:]):
                    outcome = (results.payoff_matrix[i][j],
                               results.payoff_matrix[j][i])
                    interactions.append(
                        Interaction(players=[str(p1) + str(i),
                                             str(p2) + str(j+i+1)],
                                    outcomes=score_to_chess(outcome)))

            player_names = [str(p) + str(i) for i, p in enumerate(players)]
            ratings = bayeselo(player_names, interactions, elos)

            ranked_players = sorted(
                zip(ratings, players),
                key=lambda x: x[0].mu,
                reverse=True
            )
            ranked_players = [player for elo, player in ranked_players]
            ratings.sort(key=lambda x: x.mu, reverse=True)
            ratings = [EloRate(round(r.mu), 0) for r in ratings]

            print(f"Round {gen}")
            for x, y in zip(ranked_players, ratings):
                print(f"{str(x):>15}: {y.mu}")
            print()

            first = results.ranking[0]
            last = results.ranking[-1]

            pop.checkout(branches[first])
            branches[last] = pop.branch(str(players[first]), auto_rename=True)

            players[last] = players[first]

            for p, b in zip(players, branches):
                pop.checkout(b)
                pop.commit(p)
