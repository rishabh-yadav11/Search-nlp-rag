"""Single definition of the Jaccard helper shared by search and eval.

``jaccard`` had independent copies in the /search read path (``app/diversity.py``) and the chat
eval runner (``scripts/eval_runner.py``), plus a third ratio inlined in
``_pairwise_source_jaccard`` that called neither; the eval copy had lost the empty-set guard.
Both of its call sites happened to pre-filter empty sets, so it never divided by zero -- a
correctness that rested on each caller remembering a check the helper should own. This module is
the single definition, mirroring the convention in ``scripts/_common.py``.

Only ``jaccard`` moved here. The two stopword lists were deliberately left where they are, and
that decision is load-bearing rather than tidiness: they were written for different consumers
(app/main.py localizes a body region by matching query content words; the eval runner uses its
list to make paraphrases of one question collide), and reconciling them is a behaviour change
wearing a de-duplication's clothes. Widening the eval list feeds ``group_prompts``, and since
Jaccard rises as tokens are removed it would merge prompt pairs that were previously separate,
changing the reported cross-variation number without changing what the model answered or
cited -- a false merge. Widening the app list changes ``_query_content_tokens`` on the live
/search chat path, so ``"list names"`` and ``"tell me"`` would reduce to zero content tokens
and short-circuit ``body_rescue``. Both need their own change, tests and sign-off.
"""


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when either side is empty.

    The empty-set guard is part of the contract, not a convenience: callers build token sets
    from free text, so a title with no usable tokens is ordinary input and must score as
    maximally dissimilar rather than divide by zero.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
