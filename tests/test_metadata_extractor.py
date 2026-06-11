"""
Unit tests for src/metadata_extractor.py

Tests cover:
- extract_metadata() on JPEG (with EXIF) vs PNG (no EXIF)
- analyze_metadata() output keys and logic
- extract_metadata_features() feature dict structure
- metadata_score() range and scoring logic
- Error handling
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metadata_extractor import (
    extract_metadata,
    analyze_metadata,
    extract_metadata_features,
    metadata_score,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
REAL_IMG = os.path.join(DATA_DIR, 'real', 'real_001.jpg')       # JPEG with EXIF
AI_IMG = os.path.join(DATA_DIR, 'ai_generated', 'ai_gemini_img_001.png')  # PNG, no EXIF


# =====================================================================
# extract_metadata
# =====================================================================

class TestExtractMetadata:
    """Tests for extract_metadata()."""

    def test_returns_dict(self):
        meta = extract_metadata(REAL_IMG)
        assert isinstance(meta, dict)

    def test_png_returns_empty_or_dict(self):
        """PNG files typically have no EXIF, should return empty dict."""
        meta = extract_metadata(AI_IMG)
        assert isinstance(meta, dict)
        # AI PNG should have no or very few EXIF tags
        assert len(meta) <= 5  # lenient — some PNGs may have a few tags


# =====================================================================
# analyze_metadata
# =====================================================================

class TestAnalyzeMetadata:
    """Tests for analyze_metadata()."""

    EXPECTED_KEYS = {
        'has_camera_data', 'has_gps', 'has_timestamps', 'has_software',
        'camera_tags_found', 'context_tags_found', 'software_info', 'tag_count',
    }

    def test_output_keys_with_metadata(self):
        meta = extract_metadata(REAL_IMG)
        analysis = analyze_metadata(meta)
        assert analysis.keys() == self.EXPECTED_KEYS

    def test_output_keys_empty_metadata(self):
        analysis = analyze_metadata({})
        assert analysis.keys() == self.EXPECTED_KEYS

    def test_empty_metadata_analysis(self):
        analysis = analyze_metadata({})
        assert analysis['has_camera_data'] is False
        assert analysis['has_gps'] is False
        assert analysis['tag_count'] == 0
        assert analysis['camera_tags_found'] == []

    def test_camera_tag_detection(self):
        """If metadata contains 'Make', it should flag has_camera_data."""
        analysis = analyze_metadata({'Make': 'Canon', 'Model': 'EOS 5D'})
        assert analysis['has_camera_data'] is True
        assert 'Make' in analysis['camera_tags_found']

    def test_software_detection(self):
        analysis = analyze_metadata({'Software': 'Adobe Photoshop'})
        assert analysis['has_software'] is True
        assert analysis['software_info'] == 'Adobe Photoshop'


# =====================================================================
# extract_metadata_features
# =====================================================================

class TestExtractMetadataFeatures:
    """Tests for extract_metadata_features()."""

    EXPECTED_KEYS = {
        'meta_tag_count', 'meta_camera_tags', 'meta_has_camera',
        'meta_has_gps', 'meta_has_timestamps', 'meta_has_software',
    }

    def test_returns_dict(self):
        features = extract_metadata_features(REAL_IMG)
        assert isinstance(features, dict)

    def test_expected_keys(self):
        features = extract_metadata_features(REAL_IMG)
        assert features.keys() == self.EXPECTED_KEYS

    def test_all_values_are_float(self):
        features = extract_metadata_features(REAL_IMG)
        for key, val in features.items():
            assert isinstance(val, float), f"{key} is {type(val)}"

    def test_binary_features_are_0_or_1(self):
        features = extract_metadata_features(REAL_IMG)
        for key in ['meta_has_camera', 'meta_has_gps', 'meta_has_timestamps', 'meta_has_software']:
            assert features[key] in (0.0, 1.0), f"{key} = {features[key]}"


# =====================================================================
# metadata_score
# =====================================================================

class TestMetadataScore:
    """Tests for metadata_score()."""

    def test_returns_float(self):
        score = metadata_score(REAL_IMG)
        assert isinstance(score, float)

    def test_score_range_real(self):
        score = metadata_score(REAL_IMG)
        assert 0.0 <= score <= 1.0

    def test_score_range_ai(self):
        score = metadata_score(AI_IMG)
        assert 0.0 <= score <= 1.0

    def test_ai_image_scores_higher(self):
        """AI images (no EXIF) should generally score higher than real images (with EXIF)."""
        real_score = metadata_score(REAL_IMG)
        ai_score = metadata_score(AI_IMG)
        # AI should score higher (more suspicious) — this is the intended design
        assert ai_score >= real_score, (
            f"Expected AI score ({ai_score}) >= real score ({real_score})"
        )

    def test_deterministic(self):
        s1 = metadata_score(REAL_IMG)
        s2 = metadata_score(REAL_IMG)
        assert s1 == s2


# =====================================================================
# Error handling
# =====================================================================

class TestMetadataErrors:
    """Tests for error handling."""

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            metadata_score('/nonexistent/image.jpg')

    def test_unsupported_extension(self):
        tmp_path = '/tmp/test_meta_bad.txt'
        with open(tmp_path, 'w') as f:
            f.write("not an image")
        try:
            with pytest.raises(ValueError):
                metadata_score(tmp_path)
        finally:
            os.remove(tmp_path)
