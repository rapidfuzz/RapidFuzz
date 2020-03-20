import _rapidfuzz_cpp.process
from typing import Iterable

def extract(query: str, choices: Iterable, limit: int = 5, score_cutoff: int = 0):
	"""
	Find all matches with a ratio above score_cutoff
	"""
	return _rapidfuzz_cpp.process.extract(query, list(choices), limit, score_cutoff)



def extractOne(query: str, choices: Iterable, score_cutoff: int = 0):
	"""
	Find the best match in a list of matches
	"""
	return _rapidfuzz_cpp.process.extractOne(query, list(choices), score_cutoff)
