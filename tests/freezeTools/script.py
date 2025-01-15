import rapidfuzz
from rapidfuzz.distance import metrics_py
from rapidfuzz.distance import metrics_cpp
rapidfuzz.distance.Levenshtein.distance('test', 'teste')
metrics_py.levenshtein_distance('test', 'teste')
metrics_cpp.levenshtein_distance('test', 'teste')