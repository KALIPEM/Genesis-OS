"""Example training script for the SpinNetwork."""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.spin_network import SpinNetwork


def main() -> None:
    input_dim = 10
    hidden_dims = [16, 16]
    output_dim = 1
    model = SpinNetwork(input_dim, hidden_dims, output_dim)

    # Dummy regression dataset: y = sum(x)
    num_samples = 1024
    x = torch.randn(num_samples, input_dim)
    y = x.sum(dim=1, keepdim=True)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()
