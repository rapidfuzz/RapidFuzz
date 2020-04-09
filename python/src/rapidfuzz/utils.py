from rapidfuzz._utils import *
import re

def default_process(sentence: str):
    alnum_re = re.compile(r"(?ui)\W")
    return alnum_re.sub(" ", sentence).strip().lower()