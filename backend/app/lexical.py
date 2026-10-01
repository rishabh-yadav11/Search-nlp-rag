"""Single definition of the Jaccard helper shared by search and eval.

``jaccard`` existed as two independent copies (the /search read path and the
chat eval runner) plus a third ratio inlined in ``_pairwise_source_jaccard``,
and the eval runner's copy had lost the empty-set guard. This module is the
single definition, mirroring the convention in ``scripts/_common.py``.

Only ``jaccard`` lives here. The two stopword lists in ``app/main.py`` and
``scripts/eval_runner.py`` were deliberately left where they are, and that
decision is load-bearing rather than tidiness:

* They serve different consumers: the app list localizes a body region by
  matching query content words against body text, the eval list makes
  paraphrases of one question collide.
* Reconciling them is not a refactor. The eval list feeds ``group_prompts``,
  and Jaccard rises monotonically as tokens are removed, so adopting the wider
  list merges prompt pairs that were previously separate -- changing the
  reported cross-variation consistency number without any change in what the
  model answered or cited. Merging is a false merge.
* The app list feeds ``_query_content_tokens`` on the live ``/search`` chat
  path, so widening it changes what a user sees (``"list names"`` and
  ``"tell me"`` reduce to zero content tokens, short-circuiting
  ``body_rescue``).

Both halves are behaviour changes wearing a de-duplication's clothes, so they
need their own change, tests and sign-off.
"""


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when either side is empty.

    The empty-set guard is part of the contract, not a convenience: token sets
    come from free text, so a title with no usable tokens is ordinary input and
    must score as maximally dissimilar rather than divide by zero.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
