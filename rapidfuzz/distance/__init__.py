try:
    from ._initialize_cpp import Editops, Opcodes, ScoreAlignment
except ImportError:
    from ._initialize_py import ScoreAlignment

from . import Hamming, Indel, Jaro, JaroWinkler, Levenshtein, LCSseq
