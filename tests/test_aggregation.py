import pytest

from alpr.aggregation import STRATEGIES, aggregate, char_vote, max_confidence, string_vote


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_no_reads_gives_no_consensus(strategy):
    assert aggregate([], strategy) is None


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_single_read_is_returned_as_is(strategy):
    consensus = aggregate([("BXYZ123", 0.8)], strategy)
    assert consensus.text == "BXYZ123"
    assert consensus.n_reads == 1


def test_voting_overrules_one_confident_outlier():
    reads = [("BXYZ123", 0.6)] * 5 + [("BXYZ128", 0.95)]
    # The original behavior: trust the single most confident frame.
    assert max_confidence(reads).text == "BXYZ128"
    vote = string_vote(reads)
    assert vote.text == "BXYZ123"
    assert vote.confidence == pytest.approx(3.0 / 3.95)
    assert vote.n_reads == 6


def test_votes_are_weighted_by_confidence():
    reads = [("AAAA111", 0.2), ("AAAA111", 0.2), ("BBBB222", 0.9)]
    assert string_vote(reads).text == "BBBB222"
    assert char_vote(reads).text == "BBBB222"


def test_char_vote_recovers_a_plate_no_single_frame_read_correctly():
    reads = [("BXYZ128", 0.9), ("BXVZ123", 0.9), ("8XYZ123", 0.9)]
    assert char_vote(reads).text == "BXYZ123"
    # Whole-string voting cannot: all three candidates are wrong.
    assert string_vote(reads).text != "BXYZ123"


def test_char_vote_confidence_is_mean_winning_share():
    reads = [("AB", 1.0), ("AB", 1.0), ("AC", 1.0), ("AB", 1.0)]
    consensus = char_vote(reads)
    assert consensus.text == "AB"
    assert consensus.confidence == pytest.approx((1.0 + 0.75) / 2)


def test_char_vote_ignores_reads_of_a_minority_length():
    reads = [("BXYZ123", 0.7), ("BXYZ123", 0.7), ("XYZ123", 0.9)]
    assert char_vote(reads).text == "BXYZ123"


def test_ties_are_broken_deterministically():
    reads = [("BBBB222", 0.5), ("AAAA111", 0.5)]
    assert string_vote(reads).text == "AAAA111"
    assert string_vote(reads[::-1]).text == "AAAA111"
    assert max_confidence(reads).text == max_confidence(reads[::-1]).text == "AAAA111"


def test_zero_confidences_do_not_divide_by_zero():
    reads = [("BXYZ123", 0.0), ("BXYZ123", 0.0), ("BXYZ128", 0.0)]
    assert string_vote(reads).text == "BXYZ123"
    assert char_vote(reads).text == "BXYZ123"


def test_unknown_strategy():
    with pytest.raises(ValueError, match="unknown aggregation strategy"):
        aggregate([("BXYZ123", 0.9)], "coin_flip")
