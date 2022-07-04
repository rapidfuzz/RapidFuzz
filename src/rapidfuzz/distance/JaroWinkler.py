try:
    from jarowinkler import jarowinkler_similarity as similarity
except ImportError:

    def similarity(s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None):
        raise NotImplementedError
