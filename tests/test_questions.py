from collections import Counter

from poison_eval.poison import MAX_POISON, make_poison_docs
from poison_eval.questions import build_questions


def test_saved_questions_match_a_fresh_build(store, questions):
    # data/questions.jsonl is reproducible from the snapshot and seed
    assert [q.__dict__ for q in build_questions(store)] == [q.__dict__ for q in questions]


def test_balanced_and_unique(questions):
    assert Counter(q.qtype for q in questions) == {"cvss": 15, "in_kev": 15, "ransomware": 15, "vendor": 15}
    assert len({q.cve_id for q in questions}) == len(questions)
    for t in ("in_kev", "ransomware"):
        golds = Counter(q.gold for q in questions if q.qtype == t)
        assert abs(golds["yes"] - golds["no"]) <= 1


def test_gold_answers_come_from_the_snapshot(store, questions):
    for q in questions:
        assert q.gold != q.poison
        if q.qtype == "cvss":
            assert float(q.gold) == store.cves[q.cve_id]["cvss"]["base_score"]
        elif q.qtype == "in_kev":
            assert (q.gold == "yes") == (q.cve_id in store.kev)
        elif q.qtype == "ransomware":
            assert (q.gold == "yes") == (store.kev[q.cve_id]["known_ransomware_use"] == "Known")
        else:
            assert q.gold == store.kev[q.cve_id]["vendor"]
            assert not any(q.poison.lower() in a.lower() for a in q.gold_aliases)


def test_poison_docs_state_the_false_claim(questions):
    for q in questions:
        docs = make_poison_docs(q, MAX_POISON)
        assert len({d.source_id for d in docs}) == MAX_POISON
        for d in docs:
            assert d.source_id.startswith("web:") and q.cve_id in d.text
            if q.qtype in ("cvss", "vendor"):
                assert q.poison in d.text
