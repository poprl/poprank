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


def payoff_to_interac(players, payoff):
    interactions = []

    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players[i+1:]):

            outcome = (payoff[i][j], payoff[j][i])
            outcome = score_to_chess(outcome)

            interac = Interaction(
                players=[str(p1) + str(i), str(p2) + str(j+i+1)],
                outcomes=outcome)

            interactions.append(interac)

    return interactions


class TestAxelrodImplementation(unittest.TestCase):

    def test_axelrod_bayeselo(self):

        # Create initial population

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

        # Add the initial agents to the population

        pop = Population()
        branches = []
        for p in players:
            b = pop.branch(str(p), auto_rename=True)
            pop.checkout(b)
            pop.commit(p)
            branches.append(b)

        for gen in range(7):

            # Rank current agents

            elos = [EloRate() for _ in players]

            tournament = axl.Tournament(players)
            results = tournament.play(progress_bar=False)

            interactions = payoff_to_interac(players, results.payoff_matrix)

            player_names = [str(p) + str(i) for i, p in enumerate(players)]
            ratings = bayeselo(player_names, interactions, elos)

            ratings, players, branches = [list(t) for t in zip(*sorted(
                zip(ratings, players, branches),
                key=lambda x: x[0].mu,
                reverse=True
            ))]

            print(f"Round {gen}")
            for x, y in zip(ratings, players):
                print(f"{str(y):>15}: {x.mu}")
            print()

            # Remove the worst player and replace with a copy of the best

            pop.checkout(branches[0])
            branches[-1] = pop.branch(str(players[0]), auto_rename=True)

            players[-1] = players[0]

            for p, b in zip(players, branches):
                pop.checkout(b)
                pop.commit(p)
