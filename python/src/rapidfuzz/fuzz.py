import _rapidfuzz_cpp.fuzz as fuzz_cpp


def ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    calculates a simple ratio between two strings

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.ratio("this is a test", "this is a test!")
        96.55171966552734
    """
    return fuzz_cpp.ratio(s1, s2, score_cutoff, preprocess)


def partial_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    calculates a partial ratio between two strings

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.partial_ratio("this is a test", "this is a test!")
        100.0
    """
    return fuzz_cpp.partial_ratio(s1, s2, score_cutoff, preprocess)


def token_sort_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    sorts the words in the string and calculates the fuzz.ratio between them

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
        100.0
    """
    return fuzz_cpp.token_sort_ratio(s1, s2, score_cutoff, preprocess)


def partial_token_sort_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    sorts the words in the strings and calculates the fuzz.partial_ratio between them

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """
    return fuzz_cpp.partial_token_sort_ratio(s1, s2, score_cutoff, preprocess)


def token_set_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    Compares the words in the strings based on unique and common words between them using fuzz.ratio

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
        83.8709716796875
        >>> fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
        100.0
    """
    return fuzz_cpp.token_set_ratio(s1, s2, score_cutoff, preprocess)


def partial_token_set_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    Compares the words in the strings based on unique and common words between them using fuzz.partial_ratio

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """
    return fuzz_cpp.partial_token_set_ratio(s1, s2, score_cutoff, preprocess)


def token_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    Helper method that returns the maximum of fuzz.token_set_ratio and fuzz.token_sort_ratio
        (faster than manually executing the two functions)

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """
    return fuzz_cpp.token_ratio(s1, s2, score_cutoff, preprocess)


def partial_token_ratio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    Helper method that returns the maximum of fuzz.partial_token_set_ratio and fuzz.partial_token_sort_ratio
        (faster than manually executing the two functions)

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """
    return fuzz_cpp.partial_token_ratio(s1, s2, score_cutoff, preprocess)


def QRatio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
	"""
	calculates a quick ratio between two strings using fuzz.ratio

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100

    Example:
        >>> fuzz.ratio("this is a test", "this is a test!")
        96.55171966552734
	"""
	return fuzz_cpp.ratio(s1, s2, score_cutoff, preprocess)


def WRatio(s1: str, s2: str, score_cutoff: float = 0, preprocess: bool = True):
    """
    Calculates a weighted ratio based on the other ratio algorithms

    Args:
        s1 (str): first string to compare
        s2 (str): second string to compare
        score_cutoff (float): Optional argument for a score threshold as a float between 0 and 100.
            For ratio < score_cutoff 0 is returned instead of the ratio. Defaults to 0.
        preprocess (bool): Optional argument to specify whether the strings should be preprocessed 
            using utils.default_process. Defaults to True.

    Returns:
        float: ratio between s1 and s2 as a float between 0 and 100
    """
    return fuzz_cpp.WRatio(s1, s2, score_cutoff, preprocess)