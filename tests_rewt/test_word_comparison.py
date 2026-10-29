import numpy as np
import pytest
from scripts.analyze_rewt_word_refinement import paired_interval


def test_paired_bootstrap_keeps_constant_seed_document_differences():
    result = paired_interval(np.full((3,25), -.125), repetitions=500)
    assert result['mean'] == -.125
    assert result['ci95'] == [-.125,-.125]
    assert result['seed_means'] == [-.125]*3


def test_bootstrap_preserves_both_signs_when_seed_effects_disagree():
    values = np.repeat(np.array([-.1,0,.1])[:,None],25,axis=1)
    result = paired_interval(values,repetitions=500)
    assert result['mean'] == pytest.approx(0)
    assert result['ci95'][0] < 0 < result['ci95'][1]
