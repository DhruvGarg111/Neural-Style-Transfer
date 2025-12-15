import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import os

from transformer_net import TransformerNet

def stylize_func(content_image, style_choice):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if style_choice == "Style 1":
        model_path = "ckpt_epoch_0_step_12000.pth"
    elif style_choice == "Style 2":
        model_path = "dark_asthetic_final.pth"
    elif style_choice == "Style 3":
        model_path = "candy_ckpt_epoch_0_step_36400.pth"
    elif style_choice == "Style 4":
       model_path = "mosaic_ckpt_epoch_1_step_74000.pth" 
    else:
        return None 

    if not os.path.exists(model_path):
        raise gr.Error(f"Model file {model_path} not found! Please upload it to the Space.")

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()

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
                "Style 1",
                "Style 2",
                "Style 3",
                "Style 4",
            ],
            label="Select Style",
            value="Style 1",
        ),
    ],
    outputs=gr.Image(type="pil", label="Stylized Output"),
    title=title,
    description=description
)

if __name__ == "__main__":
    interface.launch()