from byteprof.counting import first_divergence


def test_identical():
    assert first_divergence(b"hello", b"hello") is None
    assert first_divergence(b"", b"") is None


def test_diverges_at_zero():
    assert first_divergence(b"abc", b"xbc") == 0


def test_diverges_midway():
    assert first_divergence(b"abcdef", b"abcXef") == 3


def test_different_lengths():
    assert first_divergence(b"abc", b"abcd") == 3  # prefix: diverges where the shorter ends
    assert first_divergence(b"abcd", b"abc") == 3
    assert first_divergence(b"", b"a") == 0
    assert first_divergence(b"abx", b"abcd") == 2  # a real mismatch wins over the length difference
