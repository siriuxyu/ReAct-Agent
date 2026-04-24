import pytest


def test_token_f1_exact_match():
    from benchmark.benchmark_runner import compute_token_f1
    assert compute_token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_token_f1_no_overlap():
    from benchmark.benchmark_runner import compute_token_f1
    assert compute_token_f1("apple banana", "orange grape") == pytest.approx(0.0)


def test_token_f1_partial_overlap():
    from benchmark.benchmark_runner import compute_token_f1
    score = compute_token_f1("the cat sat on mat", "the cat")
    # prediction tokens: {the:1, cat:1, sat:1, on:1, mat:1}
    # reference tokens:  {the:1, cat:1}
    # common = 2, precision = 2/5 = 0.4, recall = 2/2 = 1.0
    # F1 = 2*0.4*1/(0.4+1) = 0.571...
    assert 0.5 < score < 0.65


def test_token_f1_empty_prediction():
    from benchmark.benchmark_runner import compute_token_f1
    assert compute_token_f1("", "the cat sat") == pytest.approx(0.0)


def test_token_f1_empty_reference():
    from benchmark.benchmark_runner import compute_token_f1
    assert compute_token_f1("the cat", "") == pytest.approx(0.0)


def test_token_f1_repeated_tokens():
    from benchmark.benchmark_runner import compute_token_f1
    # pred: {cat:3}, ref: {cat:1} → common=1, precision=1/3, recall=1/1, F1=0.5
    score = compute_token_f1("cat cat cat", "cat")
    assert score == pytest.approx(0.5)
