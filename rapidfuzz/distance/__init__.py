try:
    from ._initialize_cpp import Editop, Editops, Opcode, Opcodes, ScoreAlignment
except ImportError:
    from ._initialize_py import Editop, Editops, Opcode, Opcodes, ScoreAlignment

from . import Hamming, Indel, Jaro, JaroWinkler, Levenshtein, LCSseq
