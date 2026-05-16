import numpy as np

from research.sonoedit_qwen3_tts.targets import extract_codec0_target, extract_codec_target, select_codec0, select_codec_codes, slice_target_span


def test_select_codec0_from_qwen_16_codebooks():
    codes = np.arange(16 * 5).reshape(16, 5)

    np.testing.assert_array_equal(select_codec0(codes), codes[0])
    np.testing.assert_array_equal(select_codec_codes(codes), codes.T)


def test_select_codec0_from_qwen_tokenizer_model_output():
    class Encoded:
        def __init__(self):
            self.audio_codes = [np.arange(5 * 16).reshape(5, 16)]

    np.testing.assert_array_equal(select_codec0(Encoded()), np.array([0, 16, 32, 48, 64]))


def test_manual_target_frame_span():
    codec0 = np.array([10, 11, 12, 13, 14])

    np.testing.assert_array_equal(slice_target_span(codec0, (1, 4)), np.array([11, 12, 13]))


def test_invalid_frame_span_fails_clearly():
    try:
        slice_target_span(np.array([1, 2]), (1, 3))
    except ValueError as exc:
        assert "invalid target frame span" in str(exc)
    else:
        raise AssertionError("expected invalid span to fail")


def test_tokenizer_codec0_extraction_uses_explicit_span(tmp_path):
    class FakeTokenizer:
        def encode_audio(self, path):
            assert path.endswith("target.wav")
            return np.arange(16 * 6).reshape(16, 6)

    target = extract_codec0_target(FakeTokenizer(), tmp_path / "target.wav", (2, 5))

    np.testing.assert_array_equal(target, np.array([2, 3, 4]))


def test_tokenizer_codec0_extraction_uses_qwen_encode_method(tmp_path):
    class FakeTokenizer:
        def encode(self, path):
            assert path.endswith("target.wav")
            return {"audio_codes": [np.arange(6 * 16).reshape(6, 16)]}

    target = extract_codec0_target(FakeTokenizer(), tmp_path / "target.wav", (2, 5))

    np.testing.assert_array_equal(target, np.array([32, 48, 64]))


def test_tokenizer_all_codebook_extraction_uses_frame_span(tmp_path):
    class FakeTokenizer:
        def encode(self, path):
            assert path.endswith("target.wav")
            return {"audio_codes": [np.arange(6 * 16).reshape(6, 16)]}

    target = extract_codec_target(FakeTokenizer(), tmp_path / "target.wav", (2, 5))

    np.testing.assert_array_equal(target, np.arange(6 * 16).reshape(6, 16)[2:5])
