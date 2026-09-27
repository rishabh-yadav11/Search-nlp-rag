"""Every knob in app/config.py must be read by something, or not exist at all.

A setting that nothing reads is a false promise to whoever deploys this app:
it is defined in config.py, it is listed in .env.example, an operator sets it,
and no behaviour moves. Issue #263 shipped two such knobs -- a profile-decay
lambda that was never wired to the hardcoded 30-day decay, and a candidate-pool
limit the recommender never consulted.

So this module makes the class of defect fail the suite instead. A knob counts
as read when it is accessed as ``config.NAME`` / ``Config.NAME``, or when its
name appears as a string literal, which is how the rate-limit knobs are
resolved (``public_rate_limit("click", "PUBLIC_CLICK_RATE_PER_MIN")`` reads the
attribute name out of a string and passes it to ``getattr``).

To keep a knob that is deliberately inert, add it to INTENTIONALLY_INERT with a
reason. The reason is required: an allowlist entry with no explanation is just
the bug again, one level up.
"""
import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CONFIG_PY = BACKEND / "app" / "config.py"
THIS_FILE = Path(__file__).resolve()

_SKIP_DIRS = {"venv", ".git", "node_modules", "__pycache__", "build", "dist", "data"}

# Config knobs that nothing reads on purpose. Keep the list empty if you can:
# every entry here is an operator-visible setting with no effect.
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
    """Every shipped Python file, minus this test and config.py itself.

    config.py is excluded because it mentions every knob in its own
    ``os.getenv`` call, and this test because its allowlist spells knob names out
    in plain text -- either would make every knob look referenced.
    """
    files = [
        path
        for path in BACKEND.rglob("*.py")
        if not _SKIP_DIRS.intersection(path.parts)
        and path.resolve() not in (CONFIG_PY.resolve(), THIS_FILE)
    ]
    assert files, "no python files found -- the scan is not looking anywhere"
    return files


def _is_referenced(knob: str, corpus: str) -> bool:
    """Whether anything outside config.py reads ``knob``."""
    if re.search(rf"\b(?:config|Config)\.{knob}\b", corpus):
        return True
    # Attribute access through a string, e.g. getattr(config, name) where the
    # caller was handed the knob name as a literal.
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
    """An allowlist entry must name a real knob and justify itself.

    Without this, deleting a knob would leave its allowlist entry behind and
    the entry would silently outlive the reason for it.
    """
    knobs = _config_knobs()
    missing = sorted(set(INTENTIONALLY_INERT) - set(knobs))
    assert not missing, f"INTENTIONALLY_INERT names knobs config.py no longer defines: {missing}"

    unjustified = sorted(name for name, reason in INTENTIONALLY_INERT.items() if not reason.strip())
    assert not unjustified, f"INTENTIONALLY_INERT entries need a reason: {unjustified}"


def test_scan_actually_sees_knob_references():
    """Guard on the guard: a corpus that found nothing would pass vacuously.

    A scanner that silently stopped matching (renamed module, typo in the
    pattern, backend moved) would make the test above pass with every knob
    unreadable.
    """
    corpus = "\n".join(path.read_text() for path in _source_files())
    known_reader = "RECOMMEND_DEFAULT_LIMIT"
    assert known_reader in _config_knobs(), f"{known_reader} should still be a knob"
    assert _is_referenced(known_reader, corpus), (
        f"{known_reader} is read by app/recommender.py, so the scan must find it; "
        "the reference scan is broken and test_every_config_knob_is_read_somewhere "
        "is passing vacuously"
    )
