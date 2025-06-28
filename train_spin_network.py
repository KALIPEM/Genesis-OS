import torch
import torch.nn as nn
import torch.optim as optim

class SpinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))

def train(epochs: int = 100):
    net = SpinNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(epochs):
        inputs = torch.randn(5, 1)
        target = torch.sin(inputs)

        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    print("Training complete!")

if __name__ == "__main__":
    train()
