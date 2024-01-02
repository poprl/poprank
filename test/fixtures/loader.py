import json
from os.path import dirname


def load_fixture(name: str):
    d = dirname(__file__)
    filepath = f"{d}/{name}.json"

    with open(filepath, "r") as f:
        fixture = json.load(f)

    return fixture
