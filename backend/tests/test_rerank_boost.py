from dataclasses import dataclass

from app.rerank_boost import (
    BOOST_SUMMARY,
    BOOST_TITLE,
    apply_entity_boost,
    extract_entities,
)


@dataclass
class FakeResult:
    id: int
    title: str
    score: float
    summary: str = ""


def _res(id: int, title: str, score: float, summary: str = "") -> FakeResult:
    return FakeResult(id=id, title=title, score=score, summary=summary)


def test_extract_entities_brands_and_caps():
    assert extract_entities("who acquired Housing.com") == ["housing.com"]
    assert "ola electric" in extract_entities("Ola Electric IPO price band")
    assert extract_entities("latest funding news") == []
    assert extract_entities("") == []


def test_title_mention_outranks_nonmention():
    hit = _res(1, "Housing.com raises $100M from investors", 0.8)
    miss = _res(2, "Flipkart closes Series J round", 0.9)
    out = apply_entity_boost("who acquired Housing.com", [miss, hit])
    assert out[0].id == hit.id


def test_summary_boost_between_title_and_plain():
    plain = _res(1, "Funding round closes", 0.9, "Generic market update")
    summary = _res(2, "Company files prospectus", 0.85, "Ola Electric's price band revealed")
    title = _res(3, "Ola Electric price band set", 0.8, "")
    out = apply_entity_boost("Ola Electric IPO price band", [plain, summary, title])
    assert [r.id for r in out] == [title.id, summary.id, plain.id]


def test_boost_multipliers():
    title = _res(1, "Nykaa expands retail", 2.0)
    summary = _res(2, "Retail expansion", 2.0, "Nykaa reported growth")
    plain = _res(3, "Retail expansion", 2.0)
    out = apply_entity_boost("Nykaa growth", [plain, summary, title])
    assert out[0].id == title.id
    assert abs(out[0].score - 2.0 * BOOST_TITLE) < 1e-9
    assert out[1].id == summary.id
    assert abs(out[1].score - 2.0 * BOOST_SUMMARY) < 1e-9
    assert out[2].id == plain.id
    assert abs(out[2].score - 2.0) < 1e-9


def test_no_entity_query_order_preserved():
    a = _res(1, "Latest funding deals", 0.7)
    b = _res(2, "New fintech round", 0.9)
    c = _res(3, "M&A activity", 0.5)
    out = apply_entity_boost("latest funding news", [a, b, c])
    assert out == [a, b, c]


def test_case_insensitive_match():
    a = _res(1, "BYJU'S raises $200M round", 0.8)
    b = _res(2, "Edtech funding activity", 0.9)
    out = apply_entity_boost("byju's funding", [b, a])
    assert out[0].id == a.id


def test_multi_entity_boosts_any():
    paytm = _res(1, "Paytm reports Q2 results", 0.8)
    ola = _res(2, "Ola Electric files for IPO", 0.85)
    plain = _res(3, "Market wrap", 0.9)
    out = apply_entity_boost("Paytm and Ola Electric funding", [plain, paytm, ola])
    assert out[0].id in (paytm.id, ola.id)
    assert out[-1].id == plain.id


def test_equal_scores_keep_input_order():
    a = _res(1, "Paytm wallet", 2.0)
    b = _res(2, "Paytm payments", 2.0)
    out = apply_entity_boost("Paytm news", [a, b])
    assert [r.id for r in out] == [a.id, b.id]


def test_input_not_mutated():
    a = _res(1, "Housing.com deal", 0.8)
    b = _res(2, "Other news", 0.7)
    lst = [a, b]
    out = apply_entity_boost("Housing.com acquisition", lst)
    assert lst == [a, b]
    assert a.score == 0.8 and b.score == 0.7
    assert out is not lst
    assert out[0] is not a
    assert out[0].score == 0.8 * BOOST_TITLE


def test_boost_constants():
    assert BOOST_TITLE == 1.25
    assert BOOST_SUMMARY == 1.10


def test_multiword_entity_kept_distinct_not_bare_headword():
    # "Banyan Netfaqs Pvt Ltd" must resolve to the single entity "banyan
    # netfaqs", never the bare token "banyan" that would conflate it with
    # "Banyan Tree Finance" / "Banyan Green".
    ents = extract_entities("What is the latest news about Banyan Netfaqs Pvt Ltd?")
    assert "banyan netfaqs" in ents
    assert "banyan" not in ents
    assert "netfaqs" not in ents
    assert "pvt" not in ents
    assert "ltd" not in ents


def test_sector_noun_not_over_expanded():
    # The generic sector phrase "consumer internet" must not emit bare tokens
    # ("consumer", "internet") that over-boost unrelated articles.
    ents = extract_entities("What is the outlook for the consumer internet sector?")
    assert "consumer" not in ents
    assert "internet" not in ents


def test_capitalized_sector_phrase_not_over_expanded():
    # Realistic capitalized input hits the multi-word run path. A leading generic
    # noun ("Consumer") and trailing generic noun ("Sector") must be stripped, and
    # the surviving bare "internet" must not be emitted as a spurious entity that
    # over-boosts every internet-sector article.
    ents = extract_entities("What is the outlook for the Consumer Internet Sector?")
    assert "consumer" not in ents
    assert "internet" not in ents
    assert "sector" not in ents


def test_suffix_only_phrase_yields_no_entity():
    # "Pvt Ltd" / "Co Ltd" leave only a bare suffix/head token after stripping and
    # must not produce spurious short entities ("pvt"/"co").
    assert "pvt" not in extract_entities("Pvt Ltd leads funding")
    assert "co" not in extract_entities("Co Ltd acquisition")


def test_known_brand_survives_subsuming_run():
    # A brand must stay even when a longer non-brand run contains it.
    ents = extract_entities("Ola Electric IPO price band")
    assert "ola electric" in ents


def test_single_unknown_capitalized_word_not_an_entity():
    # "Latest" / "Funding" alone (not part of a multi-word proper noun) are
    # common nouns, not entities.
    assert "funding" not in extract_entities("Latest funding news round")
    assert "latest" not in extract_entities("Latest funding news round")
