import pandas as pd
import pytest

from common.sorel_labels import TAG_COLS, dominant_tag_labels, label_stats


def _meta(rows):
    return pd.DataFrame(rows, columns=["sha256"] + TAG_COLS)


def test_argmax_picks_the_largest_tag_count():
    m = _meta([["a", 0, 0, 9, 1, 0, 0, 0, 0, 0, 0, 0]])
    out = dominant_tag_labels(m, TAG_COLS)
    assert out.loc[0] == "ransomware"


def test_all_zero_tag_rows_are_dropped():
    m = _meta([["a", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               ["b", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    out = dominant_tag_labels(m, TAG_COLS)
    assert list(out.index) == [1]
    assert out.loc[1] == "adware"


def test_ties_break_on_declared_column_order():
    # adware and worm tie at 5; adware is earlier in TAG_COLS, so it wins.
    row = ["a", 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0]
    out = dominant_tag_labels(_meta([row]), TAG_COLS)
    assert out.loc[0] == "adware"


def test_tie_break_is_stable_under_row_reordering():
    row = ["a", 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0]
    a = dominant_tag_labels(_meta([row]), TAG_COLS).tolist()
    b = dominant_tag_labels(_meta([row, row]), TAG_COLS).tolist()
    assert a[0] == b[0] == b[1]


def test_label_stats_counts_drops_and_ties():
    m = _meta([
        ["a", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # all-zero -> dropped
        ["b", 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],   # tie adware/worm
        ["c", 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],   # clean ransomware
    ])
    s = label_stats(m, TAG_COLS)
    assert s["total_rows"] == 3
    assert s["dropped_all_zero"] == 1
    assert s["labelled_rows"] == 2
    assert s["tied_rows"] == 1
    assert s["class_counts"]["ransomware"] == 1
    assert s["class_counts"]["adware"] == 1
