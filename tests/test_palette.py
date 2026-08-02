from common import palette


def test_datasets_never_borrow_a_model_colour():
    # The bug this guards: the same hex meant "random forest" in one figure and
    # "CIC-MalMem" in the next, so a reader who learned a colour was misled by
    # it one page later.
    model_hues = set(palette.MODEL.values())
    dataset_hues = set(palette.DATASET.values()) | set(palette.DATASET_TASK.values())
    assert model_hues.isdisjoint(dataset_hues)


def test_classical_and_quantum_families_do_not_overlap():
    assert set(palette.CLASSICAL.values()).isdisjoint(set(palette.QUANTUM.values()))


def test_kernel_colours_follow_the_framework_families():
    # A fidelity kernel is quantum and an RBF kernel is classical, so reusing
    # the framework hues here is deliberate rather than a collision.
    assert palette.KERNEL["qsvm_fidelity"] == palette.FRAMEWORK["quantum"]
    assert palette.KERNEL["rbf_control"] == palette.FRAMEWORK["classical"]


def test_every_model_has_a_marker_and_a_label():
    for key in palette.MODEL:
        assert key in palette.MARKER
        assert key in palette.LABEL


def test_neutral_is_not_reused_as_a_data_colour():
    data_hues = set(palette.MODEL.values()) | set(palette.DATASET.values())
    assert palette.NEUTRAL not in data_hues
