import sys
from array import array

try:
    from future_builtins import zip
except ImportError:
    pass

if sys.version_info >= (3,10,0):
    def popcount(x):
        return x.bit_count()
else:
    def popcount(x):
        return bin(x).count('1')

def norm_distance(dist, lensum, score_cutoff):
    ratio = 100.0 - 100.0 * dist / lensum if lensum else 100.0
    return ratio if ratio >= score_cutoff else 0

def remove_common_affix(s1, s2):
    prefix = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2:
            break
        prefix += 1

    if prefix:
        s1 = s1[prefix:]
        s2 = s2[prefix:]

    if not s1 or not s2:
        return s1, s2

    affix = 0
    for ch1, ch2 in zip(reversed(s1), reversed(s2)):
        if ch1 != ch2:
            break
        affix += 1

    if affix:
        s1 = s1[:len(s1)-affix]
        s2 = s2[:len(s2)-affix]

    return s1, s2

if sys.version_info >= (3,0,0):
    def _hash(s):
        if isinstance(s, str) and len(s) == 1:
            return ord(s[0])
        return hash(s)

    def conv_sequences(s1, s2):
        if type(s1) is type(s2):
            return s1, s2
    
        if isinstance(s1, str):
            s1 = array('Q', (ord(ch) for ch in s1))
        elif not isinstance(s1, bytes) and not isinstance(s1, array):
            s1 = array('Q', (hash(ch) for ch in s1))
    
        if isinstance(s2, str):
            s2 = array('Q', (ord(ch) for ch in s2))
        elif not isinstance(s1, bytes) and not isinstance(s1, array):
            s2 = array('Q', (hash(ch) for ch in s2))
        
        return s1, s2
else:
    def _hash(s):
        if (isinstance(s, str) or isinstance(s, unicode)) and len(s) == 1:
            return ord(s[0])
        return hash(s)

    def conv_sequences(s1, s2):
        if type(s1) is type(s2):
            return s1, s2
    
        if isinstance(s1, str) or isinstance(s1, unicode):
            s1 = array('Q', (ord(ch) for ch in s1))
        elif not isinstance(s1, array):
            s1 = array('Q', (hash(ch) for ch in s1))
    
        if isinstance(s2, str) or isinstance(s2, unicode):
            s2 = array('Q', (ord(ch) for ch in s2))
        elif not isinstance(s1, array):
            s2 = array('Q', (hash(ch) for ch in s2))
        
        return s1, s2