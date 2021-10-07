from rapidfuzz.py_string_metric import (
    normalized_indel_distance as _normalized_indel_distance
)
from rapidfuzz.details.common import (
    conv_sequences
)
from rapidfuzz.utils import default_process
from array import array


def ratio(s1, s2, processor=None, score_cutoff=None):
    """
    Calculates the normalized InDel distance.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 100

    See Also
    --------
    rapidfuzz.string_metric.normalized_levenshtein : Normalized levenshtein distance

    Notes
    -----
    .. image:: img/ratio.svg

    Examples
    --------
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
    """
    if s1 is None or s2 is None:
        return 0.0

    if s1 is s2:
        return 100.0

    score_cutoff = 0.0 if score_cutoff is None else score_cutoff
    
    if processor is True:
        s1 = default_process(s1)
        s2 = default_process(s2)
    elif processor:
        s1 = processor(s1)
        s2 = processor(s2)

    s1, s2 = conv_sequences(s1, s2)

    return _normalized_indel_distance(s1, s2, score_cutoff)
