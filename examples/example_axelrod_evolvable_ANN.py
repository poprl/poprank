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
                players=["EvolvableANN" + str(i), "EvolvableANN" + str(j+i+1)],
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


def example_axelrod_evolvable_ANN():
    """This is an example of usage for popcore and poprank tracking a single
    agent evolving through play"""

    # ----- Create initial population ----- #
    player = axl.EvolvableANN(num_features=17, num_hidden=3)

    pop = Population()                             # Instantiate the population
    b = pop.branch(str(player), auto_rename=True)  # Create a branch per player
    pop.checkout(b)                                # Move to the new branch
    pop.commit(player)                             # Commit the new player

    # A few generations of the agent playing the previous version of itself and
    # then mutating
    for gen in range(7):

        # Selfplay (here it's useless as it's not taken into account when
        # mutating the agent, but in practice you would use this
        # to calculate loss and perform SGD for example)
        match = axl.Match((player, pop.current_node.model_parameters),
                          turns=10)
        match.play()

        # Create a new version of the model
        player = player.mutate()

        # Commit the new player to the population
        pop.commit(player)

    # ----- Rate the entire lineage ----- #

    # Get the previous iterations of the player
    lineage = reversed(pop.get_commit_history()[:-1])
    players = [p.model_parameters for p in pop.get_commits(lineage)]
    names = ["EvolvableANN" + str(i) for i, p in enumerate(players)]
    elos = [EloRate() for _ in players]

    # Make them play a tournament
    interactions = generate_interactions(players)

    # Rate them
    ratings = bayeselo(names, interactions, elos)

    # Sort players based on ratings
    ratings, names = sort_from_rate(ratings, names)

    # Display results
    print("Ratings of the lineage")
    for x, y in zip(ratings, names):
        print(f"{str(y):>15}: {x.mu}")
    print()


if __name__ == "__main__":
    example_axelrod_evolvable_ANN()
