import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
'''
class FrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (train, val, or test folder).
            sequence_length (int): Number of frames in an input sequence.
            transform (callable, optional): Optional transform to be applied to frames.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = []

        # Collect all sequences
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for video_dir in os.listdir(class_path):
                video_path = os.path.join(class_path, video_dir)
                if not os.path.isdir(video_path):
                    continue
                frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
                if len(frames) >= sequence_length + 10:  # Ensure enough frames for input + target
                    self.data.append((frames[:sequence_length], frames[sequence_length:sequence_length + 10]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames, target_frames = self.data[idx]

        input_sequence = [Image.open(frame).convert('L') for frame in input_frames]
        target_sequence = [Image.open(frame).convert('L') for frame in target_frames]

        if self.transform:
            input_sequence = torch.stack([self.transform(frame) for frame in input_sequence])
            target_sequence = torch.stack([self.transform(frame) for frame in target_sequence])

        return input_sequence, target_sequence

# Transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset and DataLoaders
train_dataset = FrameDataset(root_dir="train", transform=transform)
val_dataset = FrameDataset(root_dir="val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

'''
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, seq_length, img_size):
        super(TransformerModel, self).__init__()
        self.seq_length = seq_length
        self.img_size = img_size

        # Embedding layer for flattening and projecting image patches
        self.embedding = nn.Linear(img_size * img_size, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, embed_dim))

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4),
            num_layers=num_layers,
        )

        # Decoder for predicting the next frames
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, img_size * img_size),
            nn.Sigmoid()  # Scale output to [0, 1]
        )

    def forward(self, x):
        b, t, c, h, w = x.size()  # (batch, time, channels, height, width)
        x = x.view(b, t, -1)  # Flatten images into patches: (batch, time, img_size * img_size)
        x = self.embedding(x)  # Project to embedding dimension: (batch, time, embed_dim)
        x += self.positional_encoding[:, :t, :]  # Add positional encoding

        x = x.permute(1, 0, 2)  # Transformer expects (time, batch, embed_dim)
        encoded = self.encoder(x)  # Pass through Transformer encoder
        encoded = encoded.permute(1, 0, 2)  # Back to (batch, time, embed_dim)

        decoded = self.decoder(encoded)  # Decode each timestep
        decoded = decoded.view(b, t, 1, self.img_size, self.img_size)  # Reshape to original image size
        return decoded

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(
    input_dim=1, 
    embed_dim=128, 
    num_heads=4, 
    num_layers=3, 
    seq_length=10, 
    img_size=64
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
'''
# Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)[:, -targets.size(1):]  # Predict only target frames
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")

# Save the Model
torch.save(model.state_dict(), "transformer_model.pth")
print("Transformer model saved.")



def save_frames_as_images(frames, output_dir="predicted_frames"):
    """
    Save each frame in the predicted sequence as an image.

    Args:
        frames (numpy.ndarray): Predicted frames with shape (sequence_length, height, width[, channels]).
        output_dir (str): Directory where the frames will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Normalize frames to [0, 255] and convert to uint8
    frames = (frames * 255).astype(np.uint8)

    for i, frame in enumerate(frames):
        # Handle different frame shapes
        if len(frame.shape) == 3 and frame.shape[0] == 1:  # (1, H, W) -> (H, W)
            frame = frame.squeeze(0)

        elif len(frame.shape) == 3 and frame.shape[-1] == 1:  # (H, W, 1) -> (H, W)
            frame = frame.squeeze(-1)

        elif len(frame.shape) != 2:  # Ensure final shape is (H, W)
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        # Save the frame as a .png file
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame: {frame_path}")


# Predict frames
model.eval()
with torch.no_grad():
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        predicted_frames = model(inputs)[:, -10:]  # Predict the last 10 frames
        predicted_frames = predicted_frames[0].cpu().numpy()  # Take the first batch and convert to NumPy
        print("Predicted frames shape:", predicted_frames.shape)
        break

# Save predicted frames to a folder
save_frames_as_images(predicted_frames, output_dir="predicted_frames_transformer")



# Save Model and Optimizer State
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, "transformer_checkpoint.pth")
print("Transformer checkpoint saved.")

# Load Checkpoint
checkpoint = torch.load("transformer_checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()
print("Transformer model loaded.")
'''