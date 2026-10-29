import pytest

from poison_eval.questions import DATA_DIR, load_questions, load_store


@pytest.fixture(scope="session")
def store():
    return load_store()


@pytest.fixture(scope="session")
def questions():
    return load_questions(DATA_DIR / "questions.jsonl")
