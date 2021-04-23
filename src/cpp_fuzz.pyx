# distutils: language=c++
# cython: language_level=3
# cython: binding=True

from rapidfuzz.utils import default_process
from cpp_common cimport proc_string, is_valid_string, convert_string, hash_array, hash_sequence#, conv_sequence
from array import array

cdef inline proc_string conv_sequence(seq):
    if is_valid_string(seq):
        return convert_string(seq)
    elif isinstance(seq, array):
        return hash_array(seq)
    else:
        return hash_sequence(seq)


cdef extern from "cpp_fuzz.hpp":
    double ratio_no_process(                   proc_string, proc_string, double) nogil except +
    double partial_ratio_no_process(           proc_string, proc_string, double) nogil except +
    double token_sort_ratio_no_process(        proc_string, proc_string, double) nogil except +
    double token_set_ratio_no_process(         proc_string, proc_string, double) nogil except +
    double token_ratio_no_process(             proc_string, proc_string, double) nogil except +
    double partial_token_sort_ratio_no_process(proc_string, proc_string, double) nogil except +
    double partial_token_set_ratio_no_process( proc_string, proc_string, double) nogil except +
    double partial_token_ratio_no_process(     proc_string, proc_string, double) nogil except +
    double WRatio_no_process(                  proc_string, proc_string, double) nogil except +
    double QRatio_no_process(                  proc_string, proc_string, double) nogil except +

    double ratio_default_process(                   proc_string, proc_string, double) nogil except +
    double partial_ratio_default_process(           proc_string, proc_string, double) nogil except +
    double token_sort_ratio_default_process(        proc_string, proc_string, double) nogil except +
    double token_set_ratio_default_process(         proc_string, proc_string, double) nogil except +
    double token_ratio_default_process(             proc_string, proc_string, double) nogil except +
    double partial_token_sort_ratio_default_process(proc_string, proc_string, double) nogil except +
    double partial_token_set_ratio_default_process( proc_string, proc_string, double) nogil except +
    double partial_token_ratio_default_process(     proc_string, proc_string, double) nogil except +
    double WRatio_default_process(                  proc_string, proc_string, double) nogil except +
    double QRatio_default_process(                  proc_string, proc_string, double) nogil except +


def ratio(s1, s2, processor=False, double score_cutoff=0.0):
    """
    calculates a simple ratio between two strings. This is a simple wrapper
    for string_metric.normalized_levenshtein using the weights:
    - weights = (1, 1, 2)

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
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
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/ratio.svg

    Examples
    --------
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def partial_ratio(s1, s2, processor=False, double score_cutoff=0.0):
    """
    calculates the fuzz.ratio of the optimal string alignment

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
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
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/partial_ratio.svg

    Examples
    --------
    >>> fuzz.partial_ratio("this is a test", "this is a test!")
    100.0
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return partial_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return partial_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def token_sort_ratio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    sorts the words in the strings and calculates the fuzz.ratio between them

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/token_sort_ratio.svg

    Examples
    --------
    >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
    100.0
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return token_sort_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return token_sort_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def token_set_ratio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.ratio

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/token_set_ratio.svg

    Examples
    --------
    >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    83.8709716796875
    >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
    100.0
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return token_set_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return token_set_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def token_ratio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
    (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/token_ratio.svg
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return token_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return token_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def partial_token_sort_ratio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    sorts the words in the strings and calculates the fuzz.partial_ratio between them

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/partial_token_sort_ratio.svg
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return partial_token_sort_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return partial_token_sort_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def partial_token_set_ratio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    Compares the words in the strings based on unique and common words between them
    using fuzz.partial_ratio

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/partial_token_set_ratio.svg
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return partial_token_set_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return partial_token_set_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def partial_token_ratio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    Helper method that returns the maximum of fuzz.partial_token_set_ratio and
    fuzz.partial_token_sort_ratio (faster than manually executing the two functions)

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/partial_token_ratio.svg
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return partial_token_ratio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return partial_token_ratio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def WRatio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    Calculates a weighted ratio based on the other ratio algorithms

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Notes
    -----

    .. image:: img/WRatio.svg
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return WRatio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return WRatio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)


def QRatio(s1, s2, processor=True, double score_cutoff=0.0):
    """
    Calculates a quick ratio between two strings using fuzz.ratio.
    The only difference to fuzz.ratio is, that this preprocesses
    the strings by default.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    processor: bool or callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. When processor is True ``utils.default_process``
        is used. Default is True.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 100.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    ratio : float
        ratio distance between s1 and s2 as a float between 0 and 100

    Examples
    --------
    >>> fuzz.QRatio("this is a test", "this is a test!")
    100.0
    """
    if s1 is None or s2 is None:
        return 0

    if processor is True or processor == default_process:
        return QRatio_default_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)
    elif callable(processor):
        s1 = processor(s1)
        s2 = processor(s2)

    return QRatio_no_process(conv_sequence(s1), conv_sequence(s2), score_cutoff)