rapidfuzz.process
=================

cdist
----------
.. autofunction:: rapidfuzz.process.cdist

cpdist
----------
.. autofunction:: rapidfuzz.process.cpdist

extract
-------
.. autofunction:: rapidfuzz.process.extract

extract_iter
------------
.. autofunction:: rapidfuzz.process.extract_iter

extractOne
----------
.. autofunction:: rapidfuzz.process.extractOne

Finding the matching substring
------------------------------

:func:`rapidfuzz.process.extract` returns the choice, score, and index (or mapping key), but not
alignment positions. When using :func:`rapidfuzz.fuzz.partial_ratio` as the
scorer, call :func:`rapidfuzz.fuzz.partial_ratio_alignment` on each returned
choice to obtain the matching ranges. This computes alignments only for the
selected results, and works with the default C++ implementation:

    >>> from rapidfuzz import fuzz, process
    >>> query = "needle"
    >>> choices = {
    ...     "first": "a needle in a haystack",
    ...     "second": "another needle",
    ...     "third": "unrelated",
    ... }
    >>> matches = process.extract(
    ...     query, choices, scorer=fuzz.partial_ratio,
    ...     processor=None, score_cutoff=90, limit=2,
    ... )
    >>> for choice, score, key in matches:
    ...     alignment = fuzz.partial_ratio_alignment(query, choice, processor=None)
    ...     if alignment is not None:
    ...         print(key, choice[alignment.dest_start:alignment.dest_end])
    first needle
    second needle

The ``src_start`` and ``src_end`` positions refer to the query; ``dest_start``
and ``dest_end`` refer to the choice. This recipe uses ``partial_ratio`` in
both steps: its alignment does not describe the scores of arbitrary scorers
such as the default :func:`rapidfuzz.fuzz.WRatio`.

The example disables preprocessing so that the positions can be used to slice
the original strings. If preprocessing changes a string's length or content,
the alignment positions refer to that processed string. Preprocess the query
and choices explicitly before both steps when you need to slice the processed
text; those positions cannot in general be applied to the originals.
