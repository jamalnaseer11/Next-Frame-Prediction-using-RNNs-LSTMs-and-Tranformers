import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import numpy as np

# Load Model Definitions (assuming they are defined in the provided scripts)
from models.convlstm import ConvLSTM
from models.predrnn import PredRNN
from models.trans import TransformerModel

def load_model(model_path, model, device):
    """ Load the pretrained model from a file safely """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    return model

# Initialize and load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

convlstm_model = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=3, num_layers=3).to(device)
predrnn_model = PredRNN(input_dim=1, hidden_dim=64, kernel_size=3, num_layers=3).to(device)
transformer_model = TransformerModel(input_dim=1, embed_dim=128, num_heads=4, num_layers=3, seq_length=10, img_size=64).to(device)

convlstm_model = load_model('conv_lstm_checkpoint.pth', convlstm_model, device)
predrnn_model = load_model('predrnn_checkpoint.pth', predrnn_model, device)
transformer_model = load_model('transformer_checkpoint.pth', transformer_model, device)

# Function to prepare input and predict output
def prepare_and_predict(input_dir, model, output_dir):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load and prepare the frames
    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])[:10]
    input_tensor = torch.stack([transform(Image.open(frame).convert('L')) for frame in frames]).unsqueeze(0).to(device)
    
    # Predict the next 10 frames
    with torch.no_grad():
        predicted_frames = model(input_tensor)[:, -10:]  # Get last 10 frames
        predicted_frames = predicted_frames.cpu().squeeze(0)

    # Save the predicted frames
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(predicted_frames):
        frame_np = frame.numpy().squeeze()  # Ensure it's 2D
        if frame_np.ndim > 2:
            frame_np = frame_np[0]  # Take the first channel if still more than 2D
        frame_image = Image.fromarray((frame_np * 255).astype(np.uint8), mode='L')
        frame_image.save(os.path.join(output_dir, f'frame_{i:04d}.png'))

# Directories for input frames and output
input_dir =r"test"
prepare_and_predict(input_dir, convlstm_model, 'predicted_frames_conv_lstm')
prepare_and_predict(input_dir, predrnn_model, 'predicted_frames_pred_rnn')
prepare_and_predict(input_dir, transformer_model, 'predicted_frames_transformer')

print("All frames predicted and saved.")
