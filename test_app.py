import sys
from unittest.mock import MagicMock, patch
import pytest

# Mocking dependencies before importing app
mock_torch = MagicMock()
mock_torchvision = MagicMock()
mock_gradio = MagicMock()
mock_pil = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torchvision"] = mock_torchvision
sys.modules["gradio"] = mock_gradio
sys.modules["PIL"] = mock_pil

# Define gr.Error for the test
class GradioError(Exception):
    pass

mock_gradio.Error = GradioError

# Now we can import stylize_func from app
from app import stylize_func

def test_stylize_func_none_image():
    """Test that stylize_func raises gr.Error when content_image is None."""
    with pytest.raises(GradioError, match="Please upload an input image."):
        stylize_func(None, "Style 1")

def test_stylize_func_invalid_style():
    """Test that stylize_func (via _get_model_path) raises gr.Error for invalid style."""
    with pytest.raises(GradioError, match="Invalid style 'Invalid Style'."):
        stylize_func(MagicMock(), "Invalid Style")
