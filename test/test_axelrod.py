import unittest
from popcore import Interaction
from popcore.population import Population
from poprank import EloRate
from poprank.functional import bayeselo
import axelrod as axl   # type: ignore


def score_to_chess(outcome):
    """Turns the scores into chess notation"""
    if outcome[0] > outcome[1]:
        return (1, 0)
    if outcome[0] < outcome[1]:
        return (0, 1)
    return (0.5, 0.5)


def payoff_to_interac(players, payoff):
    """Transforms a payoff matrix into a set of pairwise interactions"""
    interactions = []

    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players[i+1:]):

            outcome = (payoff[i][i+j+1], payoff[i+j+1][i])
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
            pop.checkout("_root")

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

        # ----- Rate a specific lineage ----- #

        # Get the players from a lineage
        pop.checkout(branches[0])
        lineage = pop.get_commit_history()[:-1]
        players = [p.model_parameters for p in pop.get_commits(lineage)]
        names = [str(p) + str(i) for i, p in enumerate(players)]
        elos = [EloRate() for _ in players]

        # Make them play a tournament
        tournament = axl.Tournament(players)
        results = tournament.play(progress_bar=False)
        interactions = payoff_to_interac(players, results.payoff_matrix)

        # Rate them
        ratings = ratings = bayeselo(names, interactions, elos)

        print(f"Branch {branches[0]}")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()

        # ----- Rate a specific generation ----- #

        # Get the players from the last generation
        players = [p.model_parameters for p in pop.generations[-1]]
        names = [str(p) + str(i) for i, p in enumerate(players)]
        elos = [EloRate() for _ in players]

        # Make them play a tournament
        tournament = axl.Tournament(players)
        results = tournament.play(progress_bar=False)
        interactions = payoff_to_interac(players, results.payoff_matrix)

        # Rate them
        ratings = bayeselo(names, interactions, elos)

        ratings, players = [list(t) for t in zip(*sorted(
                zip(ratings, players),
                key=lambda x: x[0].mu,
                reverse=True
        ))]

        print(f"Generation {pop.generations[-1][0].generation}")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()

        # ----- Rate all versions of all players ----- #

        pop.checkout("_root")
        lineage = pop.get_descendents()[1:]
        players = [p.model_parameters for p in pop.get_commits(lineage)]
        names = [str(p) + str(i) for i, p in enumerate(players)]
        elos = [EloRate() for _ in players]

        # Make them play a tournament
        tournament = axl.Tournament(players)
        results = tournament.play(progress_bar=False)
        interactions = payoff_to_interac(players, results.payoff_matrix)

        # Rate them
        ratings = bayeselo(names, interactions, elos)

        ratings, players = [list(t) for t in zip(*sorted(
                zip(ratings, players),
                key=lambda x: x[0].mu,
                reverse=True
        ))]

        print("Entire population")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()
