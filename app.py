import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import os
from functools import lru_cache

from transformer_net import TransformerNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STYLE_MODEL_PATHS = {
    "Style 1": "ckpt_epoch_0_step_12000.pth",
    "Style 2": "dark_asthetic_final.pth",
    "Style 3": "candy_ckpt_epoch_0_step_36400.pth",
    "Style 4": "mosaic_ckpt_epoch_1_step_74000.pth",
}
CONTENT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)


def _get_model_path(style_choice):
    if style_choice not in STYLE_MODEL_PATHS:
        available_styles = ", ".join(STYLE_MODEL_PATHS.keys())
        raise gr.Error(f"Invalid style '{style_choice}'. Choose one of: {available_styles}.")
    return STYLE_MODEL_PATHS[style_choice]


@lru_cache(maxsize=len(STYLE_MODEL_PATHS))
def _load_style_model(model_path):
    if not os.path.exists(model_path):
        raise gr.Error(f"Model file {model_path} not found! Please upload it to the Space.")

    style_model = TransformerNet()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    style_model.load_state_dict(state_dict)
    style_model.to(DEVICE)
    style_model.eval()
    return style_model


def stylize_func(content_image, style_choice):
    """Apply the selected style model to the input image."""
    if content_image is None:
        raise gr.Error("Please upload an input image.")

    model_path = _get_model_path(style_choice)
    style_model = _load_style_model(model_path)
    content_image = CONTENT_TRANSFORM(content_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = style_model(content_image).cpu()

    output = output[0].clone().clamp(0, 255).numpy()
    
    output = output.transpose(1, 2, 0).astype("uint8")
    
    stylized_image = Image.fromarray(output)
    
    return stylized_image

# --- Gradio Interface ---

title = "Neural Canvas"
description = "Upload an image and choose a style to transform it."

interface = gr.Interface(
    fn=stylize_func,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Radio(
            choices=[
                ("Vibrant impressionist", "Style 1"),
                ("Dark aesthetic", "Style 2"),
                ("Candy colors", "Style 3"),
                ("Mosaic pattern", "Style 4"),
            ],
            label="Select Style",
            info="Choose an artistic style to apply to your image.",
            value="Style 1",
        ),
    ],
    outputs=gr.Image(type="pil", label="Stylized Output"),
    title=title,
    description=description,
    api_name="stylize",
    api_description="Stylize an input image with one of the available pretrained styles."
)

if __name__ == "__main__":
    interface.launch()
