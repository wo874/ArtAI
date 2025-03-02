import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load the pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# Replace the final classification layer for art movements
num_art_movements = 6  # Number of art periods
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_art_movements)

# Load the fine-tuned weights (if available)
# model.load_state_dict(torch.load("art_movement_model.pth"))
model.eval()

# Define art periods (customize this based on your dataset or model)
art_periods = [
    "Renaissance",
    "Baroque",
    "Romanticism",
    "Impressionism",
    "Modern Art",
    "Contemporary Art"
]

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict function
def predict_image(image):
    try:
        # Convert NumPy array to PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Preprocess the image
        image = preprocess(image).unsqueeze(0)
        
        # Predict the art period
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            return art_periods[predicted.item()]
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# What Period is The Painting From?")
    gr.Markdown("Upload an image to identify its art period.")
    
    with gr.Row():
        image_input = gr.Image(label="Upload Painting")
    
    output = gr.Textbox(label="Predicted Art Period")
    
    # Predict button
    predict_button = gr.Button("Identify Art Period")
    predict_button.click(predict_image, inputs=image_input, outputs=output)

# Launch the app
demo.launch()