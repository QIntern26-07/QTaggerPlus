import json
import numpy as np
from common.ember_vectorize import PEFeatureExtractor, vectorize_rows

RAW_PATH = "data/ember/ember2018/test_features.jsonl"

import os
import pytest
pytestmark = pytest.mark.skipif(
    not os.path.exists(RAW_PATH),
    reason="EMBER raw test_features.jsonl not present (gitignored ~1.8GB); run scripts to fetch",
)


def _first_rows(n):
    rows = []
    with open(RAW_PATH) as fh:
        for _ in range(n):
            rows.append(json.loads(fh.readline()))
    return rows


def test_extractor_dim_is_2381():
    assert PEFeatureExtractor(feature_version=2).dim == 2381


def test_extractor_rejects_other_versions():
    import pytest
    with pytest.raises(ValueError):
        PEFeatureExtractor(feature_version=1)


def test_process_raw_features_shape_and_dtype():
    row = _first_rows(1)[0]
    vec = PEFeatureExtractor(feature_version=2).process_raw_features(row)
    assert vec.shape == (2381,)
    assert vec.dtype == np.float32


def test_vectorize_rows_stacks():
    vecs = vectorize_rows(_first_rows(5))
    assert vecs.shape == (5, 2381)


def test_histogram_block_matches_raw_counts():
    # First 256 dims = ByteHistogram, L1-normalized counts. Their sum must be ~1.
    row = _first_rows(1)[0]
    vec = PEFeatureExtractor(feature_version=2).process_raw_features(row)
    assert abs(vec[:256].sum() - 1.0) < 1e-4
