import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

'''class FrameDataset(Dataset):
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

import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(input_dim, hidden_dim, kernel_size))

    def forward(self, x):
        b, t, c, h, w = x.size()
        hidden_states = [layer.init_hidden(b, (h, w)) for layer in self.layers]

        outputs = []
        for t_step in range(t):
            input_tensor = x[:, t_step]
            for i, layer in enumerate(self.layers):
                hidden_states[i] = layer(input_tensor, hidden_states[i])
                input_tensor = hidden_states[i][0]
            outputs.append(input_tensor)

        outputs = torch.stack(outputs, dim=1)
        return outputs


# Model, Loss, Optimizer
'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=3, num_layers=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# Tracking training loss
train_losses = []

# Training Loop
epochs = 40
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
    train_losses.append(train_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")

# Plotting the training losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')  # Save the figure
plt.show()


import os
import cv2
import numpy as np



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
        # If the frame has unexpected shape, handle it
        if len(frame.shape) == 3 and frame.shape[0] == frame.shape[1] == frame.shape[2]:
            # Handle case where shape is (64, 64, 64)
            frame = frame.mean(axis=0)  # Collapse along the first axis to create a grayscale frame

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
save_frames_as_images(predicted_frames, output_dir="predicted_frames")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, "conv_lstm_checkpoint.pth")
print("Model and optimizer checkpoint saved.")


# Load the checkpoint
checkpoint = torch.load("conv_lstm_checkpoint.pth")

# Restore model and optimizer state
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()  # Set to evaluation mode
print("Model and optimizer checkpoint loaded.")
'''