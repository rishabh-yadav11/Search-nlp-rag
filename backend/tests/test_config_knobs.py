"""Every knob in app/config.py must be read by something, or not exist at all.

A setting nothing reads is a false promise: it is defined, it is listed in
.env.example, an operator sets it, and no behaviour moves. A knob counts as
read when it is accessed as ``config.NAME`` / ``Config.NAME`` or when its name
appears as a string literal, which is how the rate-limit knobs are resolved
(``public_rate_limit("click", "PUBLIC_CLICK_RATE_PER_MIN")`` reads the attribute
name out of a string and passes it to ``getattr``).

Deliberately NOT checked: a knob's *value*, and the effect it has. Proving
"changing this changes behaviour" is the job of a behavioural test beside the
code that reads the knob.
"""
import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CONFIG_PY = BACKEND / "app" / "config.py"
THIS_FILE = Path(__file__).resolve()

_SKIP_DIRS = {"venv", ".git", "node_modules", "__pycache__", "build", "dist", "data"}

# Deliberately unread knobs; each entry is an operator-visible setting with no effect.
INTENTIONALLY_INERT: dict[str, str] = {
    "RERANK_ONNX_DIR": (
        "Inert on purpose: the ONNX reranker backend was removed, so the exported "
        "cross-encoder cache dir is never read. The knob is kept (and kept in "
        ".env.example) so a deployed env that still sets it does not see it "
        "silently vanish; app/config.py documents the same thing inline."
    ),
}


def _config_knobs() -> dict[str, int]:
    """Class-level UPPERCASE names on config.Config, mapped to their line."""
    tree = ast.parse(CONFIG_PY.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config"]
    assert len(classes) == 1, f"expected exactly one Config class, found {len(classes)}"

    knobs: dict[str, int] = {}
    for node in classes[0].body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        for target in targets:
            if target.id.isupper():
                knobs[target.id] = target.lineno
    return knobs


def _source_files() -> list[Path]:
    """Every shipped runtime file: the app package and the CLI scripts.

    Deliberately NOT ``BACKEND.rglob("*.py")``: a test is not a reader, and
    including tests/ would let ``patch.object(config, "SOME_KNOB", ...)`` vouch
    for a knob the app ignores. config.py and this file are excluded because
    both spell every knob name out in plain text.
    """
    files = [
        path
        for root in ("app", "scripts")
        for path in (BACKEND / root).rglob("*.py")
        # Relative parts only: a checkout under a ~/build or /srv/data ancestor
        # must still be scanned.
        if not _SKIP_DIRS.intersection(path.relative_to(BACKEND).parts)
        and path.resolve() not in (CONFIG_PY.resolve(), THIS_FILE)
    ]
    assert files, "no python files found -- the scan is not looking anywhere"
    assert not any("tests" in path.parts for path in files), "tests must not count as readers"
    return files


def _is_referenced(knob: str, corpus: str) -> bool:
    """Whether anything outside config.py reads ``knob``."""
    if re.search(rf"\b(?:config|Config)\.{knob}\b", corpus):
        return True
    # Attribute access through a string, e.g. getattr(config, name).
    #
    # Deliberately loose: any standalone string literal counts, not only one
    # passed to getattr/public_rate_limit. Narrowing it to those two call shapes
    # would break whenever the rate-limit helpers change how they receive the
    # name, and a false positive on a real knob is worse than the false negative
    # left open (a knob name quoted only in a log message).
    return bool(re.search(rf"""(['"]){re.escape(knob)}\1""", corpus))


def test_every_config_knob_is_read_somewhere():
    """No knob may sit in config.py with zero readers outside it."""
    corpus = "\n".join(path.read_text() for path in _source_files())
    unread = {
        knob: line
        for knob, line in sorted(_config_knobs().items())
        if not _is_referenced(knob, corpus) and knob not in INTENTIONALLY_INERT
    }
    assert not unread, (
        "these config knobs are defined but read by nothing, so setting them "
        "does nothing (issue #263). Wire each one, delete it, or -- if it is "
        f"deliberately inert -- add it to INTENTIONALLY_INERT with a reason: {unread}"
    )


def test_inert_allowlist_is_still_accurate():
    """An allowlist entry must name a real, genuinely unread knob, with a reason.

    An entry has to PROVE it is inert, not merely claim to be -- a wired knob
    parked here would otherwise silence this test forever.
    """
    knobs = _config_knobs()
    missing = sorted(set(INTENTIONALLY_INERT) - set(knobs))
    assert not missing, f"INTENTIONALLY_INERT names knobs config.py no longer defines: {missing}"

    unjustified = sorted(name for name, reason in INTENTIONALLY_INERT.items() if not reason.strip())
    assert not unjustified, f"INTENTIONALLY_INERT entries need a reason: {unjustified}"

    corpus = "\n".join(path.read_text() for path in _source_files())
    wired = sorted(name for name in INTENTIONALLY_INERT if _is_referenced(name, corpus))
    assert not wired, (
        f"these allowlisted knobs are read by runtime code, so they are no longer "
        f"inert: remove the entry (and, if the reader is a test, wire the knob): {wired}"
    )


def test_scan_actually_sees_knob_references():
    """Guard on the guard: a corpus that found nothing would pass vacuously."""
    corpus = "\n".join(path.read_text() for path in _source_files())
    known_reader = "RECOMMEND_DEFAULT_LIMIT"
    assert known_reader in _config_knobs(), f"{known_reader} should still be a knob"
    assert _is_referenced(known_reader, corpus), (
        f"{known_reader} is read by app/recommender.py, so the scan must find it; "
        "the reference scan is broken and test_every_config_knob_is_read_somewhere "
        "is passing vacuously"
    )


def test_ranking_tuning_knobs_agree_with_the_shipped_env_template(parse_config):
    """A knob the template quotes at a value the code does not default to is a
    promise to the operator that does not hold on a fresh deploy: copying
    .env.example sets the knob to something other than the shipped default,
    silently. Same convention as the cost-budget default in test_cost_budget.py.
    """
    shipped = parse_config()
    # Exact `NAME=value` entries only: a substring match would accept
    # WEAK_RESULT_SCORE=0.35 as the shipped 0.3, the decimal drift being caught.
    template = {}
    for line in (BACKEND / ".env.example").read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            name, _, value = stripped.partition("=")
            template[name] = value
    drifted = [
        f"{name} (code {getattr(shipped, name)!r}, template {template.get(name)!r})"
        for name in (
            "RECENCY_BOOST_STRENGTH",
            "RECENCY_BOOST_DECAY_DAYS",
            "WEAK_RESULT_SCORE",
            "WEAK_RESULT_MIN_STRONG",
            "DATE_FILLER_SCORE",
        )
        if template.get(name) != str(getattr(shipped, name))
    ]
    assert not drifted, f".env.example disagrees with the shipped default of: {drifted}"
