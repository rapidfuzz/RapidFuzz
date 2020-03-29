from _rapidfuzz_cpp.fuzz import *

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
	return ratio(s1, s2, score_cutoff, preprocess)
