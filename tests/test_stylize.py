import pathlib
import sys

import gradio as gr
import numpy as np
import pytest
import torch
from PIL import Image

# == Fixtures ======================================================

@pytest.fixture(autouse=True)
def _clear_lru(monkeypatch):
    """Ensure each test starts with a cold model cache."""
    sys.modules.pop("app", None)
    yield
    # Explicitly clear the LRU cache to release model references
    try:
        import app
        app._load_style_model.cache_clear()
    except (ImportError, AttributeError):
        pass
    sys.modules.pop("app", None)


@pytest.fixture()
def dummy_model():
    """Return a tiny deterministic TransformerNet-shaped model."""
    from transformer_net import TransformerNet

    torch.manual_seed(0)
    m = TransformerNet()
    m.eval()
    return m


@pytest.fixture()
def model_ckpt(dummy_model, tmp_path, monkeypatch):
    """Save dummy model to a temp checkpoint and patch STYLE_MODEL_PATHS."""
    ckpt = tmp_path / "style.pth"
    torch.save(dummy_model.state_dict(), ckpt)
    monkeypatch.setattr("app.STYLE_MODEL_PATHS", {"TestStyle": str(ckpt)})
    return ckpt


@pytest.fixture()
def tiny_pil():
    """64×64 solid-red PIL image – small enough for fast tests."""
    return Image.new("RGB", (64, 64), color=(255, 0, 0))


# == Core inference ================================================

class TestStylize:
    """Tests for the full stylize_func round-trip."""

    def test_returns_pil_image(self, model_ckpt, tiny_pil):
        from app import stylize_func
        result = stylize_func(tiny_pil, "TestStyle")
        assert isinstance(result, Image.Image)

    def test_output_size_matches_input(self, model_ckpt, tiny_pil):
        from app import stylize_func
        result = stylize_func(tiny_pil, "TestStyle")
        assert result.size == tiny_pil.size

    def test_pixel_values_in_range(self, model_ckpt, tiny_pil):
        from app import stylize_func
        result = stylize_func(tiny_pil, "TestStyle")
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_none_image_raises(self, monkeypatch):
        monkeypatch.setattr("app.STYLE_MODEL_PATHS", {"S": "x.pth"})
        from app import stylize_func
        with pytest.raises(gr.Error):
            stylize_func(None, "S")

    def test_invalid_style_raises(self, tiny_pil, monkeypatch):
        monkeypatch.setattr("app.STYLE_MODEL_PATHS", {"A": "a.pth"})
        from app import stylize_func
        with pytest.raises(gr.Error):
            stylize_func(tiny_pil, "DOES_NOT_EXIST")


# == Project hygiene ===============================================

class TestHygiene:
    """Non-functional quality gates."""

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def test_no_crlf_in_python_files(self):
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            capture_output=True, text=True, cwd=str(self.ROOT),
        )
        for rel_path in result.stdout.strip().splitlines():
            if not rel_path:
                continue
            p = self.ROOT / rel_path
            raw = p.read_bytes()
            assert b"\r\n" not in raw, f"{p.name} has CRLF line endings"

    def test_imports_sorted(self):
        import subprocess

        pytest.importorskip("isort")
        result = subprocess.run(
            [sys.executable, "-m", "isort", "--check-only", "--diff", "--gitignore", "."],
            capture_output=True, text=True, cwd=str(self.ROOT),
        )
        assert result.returncode == 0, f"isort diff:\n{result.stdout}"

    def test_gitattributes_exists(self):
        assert (self.ROOT / ".gitattributes").exists(), (
            ".gitattributes missing — add '* text=auto' to enforce LF"
        )
