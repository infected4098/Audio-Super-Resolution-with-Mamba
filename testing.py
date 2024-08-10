import torch
import torch.nn as nn
from torch.optim import Adam

import wandb
import numpy as np
import json
from env import AttrDict, build_env


with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:
    json_config = json.load(file)
    hyperparameters = AttrDict(json_config)
audio_array = np.random.rand(32000)
wandb.init(project="example audio")
audio = wandb.Audio(audio_array, sample_rate = 16000, caption="ex")
wandb.log({"example audios": audio})
wandb.require("core")
wandb.run.name = ""
wandb.run.save()
wandb.config.update(hyperparameters)



"""class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])

        return out


# Set device to GPU if available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_size = 100
hidden_size = 2000
num_layers = 4
num_epochs = 10
learning_rate = 0.001
batch_size = 64
seq_length = 700

# Create a sample input tensor (batch_size, seq_length, input_size)
x = torch.randn(batch_size, seq_length, input_size).to(device)
y = torch.randn(batch_size, 1).to(device)  # Dummy target tensor

# Create the SimpleLSTM model and move it to the GPU
model = SimpleLSTM(input_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Dummy training loop
for epoch in range(10000):
    model.train()

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Verify CUDA usage
print("Model is on GPU:", next(model.parameters()).is_cuda)
print("Input is on GPU:", x.is_cuda)
print("Output is on GPU:", outputs.is_cuda)
print("Loss is on GPU:", loss.is_cuda)"""
