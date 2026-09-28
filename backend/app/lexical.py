"""Single definition of the Jaccard helper shared by search and eval.

``jaccard`` used to exist as two independent copies: one in the /search read
path (``app/diversity.py``) and one in the chat eval runner
(``scripts/eval_runner.py``), plus a third ratio inlined in
``_pairwise_source_jaccard`` that called neither. The eval runner's copy had
lost the empty-set guard. Both of its call sites happened to pre-filter empty
sets, so it never actually divided by zero -- but that correctness rested on
each caller remembering a check the helper should own. This module is the
single definition, mirroring the same convention as ``scripts/_common.py``.

Only ``jaccard`` moved here. The two stopword lists were deliberately left
where they are, and that decision is load-bearing rather than tidiness:

* They were written for different consumers. ``app/main.py``'s 69 words came
  with 1c083f0, which uses them to localize a body region by matching query
  content words against body text. ``scripts/eval_runner.py``'s 92 came with
  809ca37, which uses them to make paraphrases of one question collide.
* Reconciling them is not a refactor. The eval list feeds ``group_prompts``,
  and because Jaccard rises monotonically as tokens are removed, adopting the
  wider list merges prompt pairs that were previously separate -- changing the
  reported cross-variation consistency number without any change in what the
  model actually answered or cited. Merging is a false merge.
* The app list feeds ``_query_content_tokens`` on the live ``/search`` chat
  path, so widening it changes what a user sees (``"list names"`` and
  ``"tell me"`` reduce to zero content tokens, short-circuiting
  ``body_rescue``).

Both halves are behaviour changes wearing a de-duplication's clothes. They need
their own change, their own tests and explicit sign-off, so they are not
carried here.
"""


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when either side is empty.

    The empty-set guard is part of the contract, not a convenience: callers
    build token sets from free text, so a title with no usable tokens is
    ordinary input and must score as maximally dissimilar rather than divide by
    zero.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
