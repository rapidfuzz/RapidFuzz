import _rapidfuzz_cpp.process
from typing import Iterable, List, Tuple, Optional

def extract(query: str, choices: Iterable, limit: int = 5, score_cutoff: int = 0) -> List[Tuple[str, float]]:
    """ 
    Find the best matches in a list of choices

    Parameters: 
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    score_cutoff (int): Optional argument for a score threshold. Matches with
        a lower score than this number will not be returned. Defaults to 0

    Returns: 
    List[Tuple[str, float]]: returns a list of all matches that have a score > score_cutoff
  
    """
    return _rapidfuzz_cpp.process.extract(query, list(choices), limit, score_cutoff)



def extractOne(query: str, choices: Iterable, score_cutoff: int = 0) -> Optional[Tuple[str, float]]:
    """
    Find the best match in a list of choices

    Parameters: 
    query (str): string we want to find
    choices (Iterable): list of all strings the query should be compared with
    score_cutoff (int): Optional argument for a score threshold. Matches with
            a lower score than this number will not be returned. Defaults to 0

    Returns: 
    Optional[Tuple[str, float]]: returns the best match in form of a tuple or None when there is
        no match with a score > score_cutoff
    """
    return _rapidfuzz_cpp.process.extractOne(query, list(choices), score_cutoff)
