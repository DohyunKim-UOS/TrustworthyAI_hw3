# Train a small FC network on MNIST and export to ONNX
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Small FC network: 784 -> 32 -> 16 -> 10
class SmallFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))

# Load MNIST
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Train
model = SmallFC()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training...")
for epoch in range(3):
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(train_loader):.4f}")

# Test accuracy
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
print(f"Test accuracy: {correct/10000*100:.2f}%")

# Export to ONNX
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy_input, "mnist_small_fc.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11
)
print("Saved: mnist_small_fc.onnx")

# Save a sample input for verification
sample_x, sample_y = test_data[0]
np.save("sample_input.npy", sample_x.numpy())
np.save("sample_label.npy", np.array(sample_y))
print(f"Saved sample input (label={sample_y})")
