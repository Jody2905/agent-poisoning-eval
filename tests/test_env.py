import json

from poison_eval.env import Environment
from poison_eval.poison import make_poison_docs


def test_planted_docs_reach_top_five_for_every_question(store, questions):
    for q in questions:
        for level in (1, 3):
            hits = Environment(store, make_poison_docs(q, level)).search(q.question, limit=5)
            assert sum(d.doc_type == "web" for d, _, _ in hits) == level


def test_poison_does_not_touch_primary_records_or_base_index(store, questions):
    q = questions[0]
    before = len(store.index.docs)
    env = Environment(store, make_poison_docs(q, 3))
    assert len(store.index.docs) == before
    assert env.lookup_cve(q.cve_id)["source_ids"][0] == f"cve:{q.cve_id}"
    clean = Environment(store).search(q.question, limit=10)
    assert not any(d.doc_type == "web" for d, _, _ in clean)


def test_tool_calls_return_errors_instead_of_raising(store, questions):
    env = Environment(store, make_poison_docs(questions[0], 1))
    assert "error" in json.loads(env.call("lookup_cve", {"cve_id": "bad"}))
    assert "error" in json.loads(env.call("nope", {}))
    assert "error" in json.loads(env.call("lookup_cve", {"wrong_arg": 1}))
    web_id = f"web:secwatch-{questions[0].cve_id.lower()}"
    assert json.loads(env.call("get_source", {"source_id": web_id}))["found"]
    assert web_id in env.seen_sources
