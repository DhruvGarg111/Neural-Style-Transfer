import sys
from unittest.mock import MagicMock, patch, mock_open
import pytest
import os

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
import app
from app import stylize_func

def test_stylize_func_none_image():
    """Test that stylize_func raises gr.Error when content_image is None."""
    with pytest.raises(GradioError, match="Please upload an input image."):
        stylize_func(None, "Style 1")

def test_stylize_func_invalid_style():
    """Test that stylize_func (via _get_model_path) raises gr.Error for invalid style."""
    with pytest.raises(GradioError, match="Invalid style 'Invalid Style'."):
        stylize_func(MagicMock(), "Invalid Style")

def test_stylize_func_missing_model_file():
    """Test that stylize_func raises gr.Error when the style model file is missing."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(GradioError, match="Model file .* not found!"):
            stylize_func(MagicMock(), "Style 1")

@patch("app._load_style_model")
@patch("app.CONTENT_TRANSFORM")
@patch("app.Image.fromarray")
def test_stylize_func_success(mock_fromarray, mock_transform, mock_load_model):
    """Test the success path of stylize_func with all components mocked."""
    # Setup mocks
    mock_content_image = MagicMock()
    mock_style_choice = "Style 1"

    # Mock transformations and model behavior
    mock_tensor = MagicMock()
    mock_transform.return_value.unsqueeze.return_value.to.return_value = mock_tensor

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Mock model output
    mock_output_tensor = MagicMock()
    mock_model.return_value.cpu.return_value.__getitem__.return_value.clone.return_value.clamp.return_value.numpy.return_value = MagicMock()

    # Run the function
    result = stylize_func(mock_content_image, mock_style_choice)

    # Assertions
    assert result == mock_fromarray.return_value
    mock_load_model.assert_called_once()
    mock_model.assert_called_once_with(mock_tensor)
