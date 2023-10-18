from popcore import Interaction
from popcore.population import Population
from poprank import EloRate
from poprank.functional import bayeselo
import axelrod as axl   # type: ignore

# TODO: Benchmark predictive capabilities
# TODO: check visitor design pattern


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


def generate_interactions(players):
    """Play a tournament between a given set of players, and turn the results
    to a list of interactions"""

    # Instantiate and run a tournament
    tournament = axl.Tournament(players)
    results = tournament.play(progress_bar=False)

    # Turn the payoff matrix into a list of interactions
    interactions = payoff_to_interac(players, results.payoff_matrix)

    return interactions


def sort_from_rate(ratings, *lists):
    """Sorts the lists provided based on ratings"""
    return [list(t) for t in zip(*sorted(
            zip(ratings, *lists),
            key=lambda x: x[0].mu,
            reverse=True))]


def example_axelrod_bayeselo():
    """This is an example of usage for popcore and poprank"""

    # ----- Create initial population ----- #
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

    pop = Population()                            # Instantiate the population
    for p in players:
        b = pop.branch(str(p), auto_rename=True)  # Create a branch per player.
        pop.checkout(b)                           # Move to the new branch
        pop.commit(p)                             # Commit the new player
        pop.checkout("_root")                     # Return to the _root branch

    # A list of the branches for currently active players
    branches = [branch for branch in pop.branches if branch != "_root"]

    # A few generations of the popultaion playing a tournament where the worst
    # player each round is replaced by a copy of the best
    for gen in range(7):

        # Rank current players

        # Initial ratings (all default to 0 elo)
        elos = [EloRate() for _ in players]
        # List of interactions between players
        interactions = generate_interactions(players)
        # Unique str identifier for each player
        player_names = [str(p) + str(i) for i, p in enumerate(players)]

        # Calculate new ratings
        ratings = bayeselo(player_names, interactions, elos)

        # Sort players based on ratings
        ratings, players, branches = sort_from_rate(ratings, players, branches)

        # Display the players and their ratings
        print(f"Round {gen}")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()

        # Remove the worst player and replace with a copy of the best
        pop.checkout(branches[0])  # Move to the branch of the best player

        # Replace the branch of the worst player by a new branch in the list
        # of currently active branches
        branches[-1] = pop.branch(str(players[0]), auto_rename=True)

        # Replace the worst player by (effectively) a copy of the best.
        players[-1] = players[0]

        # Push the next generation of players
        for p, b in zip(players, branches):
            pop.checkout(b)
            pop.commit(p)

    # ----- Rate a specific lineage ----- #

    def rate_lineage(pop, branch):
        # Get the players from a lineage
        pop.checkout(branch)
        lineage = pop.get_commit_history()[:-1]
        players = [p.model_parameters for p in pop.get_commits(lineage)]
        names = [str(p) + str(i) for i, p in enumerate(players)]
        elos = [EloRate() for _ in players]

        # Make them play a tournament
        interactions = generate_interactions(players)

        # Rate them
        ratings = bayeselo(names, interactions, elos)

        # Sort players based on ratings
        ratings, players = sort_from_rate(ratings, players)

        # Display results
        print(f"Branch {branch}")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()

    rate_lineage(pop, branches[0])

    # ----- Rate a specific generation ----- #

    def rate_gen(pop, gen):
        # Get the players from the last generation
        players = [p.model_parameters for p in pop.generations[gen]]
        names = [str(p) + str(i) for i, p in enumerate(players)]
        elos = [EloRate() for _ in players]

        # Make them play a tournament
        interactions = generate_interactions(players)

        # Rate them
        ratings = bayeselo(names, interactions, elos)

        # Sort players based on ratings
        ratings, players = sort_from_rate(ratings, players)

        print(f"Generation {pop.generations[-1][0].generation}")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()

    rate_gen(pop, -1)

    # ----- Rate all versions of all players ----- #

    def rate_pop(pop):
        pop.checkout("_root")
        lineage = pop.get_descendents()[1:]
        players = [p.model_parameters for p in pop.get_commits(lineage)]
        names = [str(p) + str(i) for i, p in enumerate(players)]
        elos = [EloRate() for _ in players]

        # Make them play a tournament
        interactions = generate_interactions(players)

        # Rate them
        ratings = bayeselo(names, interactions, elos)

        # Sort players based on ratings
        ratings, players = sort_from_rate(ratings, players)

        # Display results
        print("Entire population")
        for x, y in zip(ratings, players):
            print(f"{str(y):>15}: {x.mu}")
        print()

    rate_pop(pop)


if __name__ == "__main__":
    example_axelrod_bayeselo()
