try:
    from jarowinkler import jaro_similarity as similarity
except ImportError:

    def similarity(s1, s2, *, processor=None, score_cutoff=None):
        raise NotImplementedError
