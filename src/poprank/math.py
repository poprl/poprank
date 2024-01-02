from math import e


def sigmoid(x: float, base: float = e, spread: float = 1.0) -> float:
    """
        TODO: Verify numerical stability
    """
    return (1.0 + base ** (x / spread)) ** -1
