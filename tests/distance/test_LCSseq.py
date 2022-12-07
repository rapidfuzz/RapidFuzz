from tests.distance.common import LCSseq


def test_basic():
    assert LCSseq.distance("", "") == 0
    assert LCSseq.distance("test", "test") == 0
    assert LCSseq.distance("aaaa", "bbbb") == 4
