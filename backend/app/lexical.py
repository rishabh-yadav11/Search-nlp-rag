


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:

    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
