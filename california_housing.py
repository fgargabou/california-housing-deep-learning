"""
California Housing Price Prediction
-----------------------------------
A reproducible regression project using PyTorch and the California Housing
dataset from scikit-learn.

The target is expressed in units of $100,000 (as provided by the dataset).
The script downloads the dataset on first run, trains a neural network,
evaluates it on a held-out test set, and saves diagnostic plots.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# 1. Configuration
# -----------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
OUTPUT_DIR = Path("outputs")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds to make the experiment more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# 2. Load and prepare the data
# -----------------------------
def load_data():
    """Load California Housing and create train/test sets without leakage."""
    dataset = fetch_california_housing()

    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # Fit the scaler only on training data to avoid data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, dataset.feature_names


# -----------------------------
# 3. Define the neural network
# -----------------------------
class HousingRegressor(nn.Module):
    """Feed-forward neural network for a regression task."""

    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------
# 4. Train the model
# -----------------------------
def train_model(model, train_loader, X_test_tensor, y_test_tensor):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []

        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_loss = criterion(test_predictions, y_test_tensor).item()
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Train MSE: {train_loss:.4f} | "
                f"Test MSE: {test_loss:.4f}"
            )

    return train_losses, test_losses


# -----------------------------
# 5. Evaluate and visualize
# -----------------------------
def evaluate_model(model, X_test_tensor, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().reshape(-1)

    actual = y_test.reshape(-1)

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    print("\nTest-set evaluation")
    print("-------------------")
    print(f"MAE :  {mae:.4f} (in $100,000 units)")
    print(f"RMSE:  {rmse:.4f} (in $100,000 units)")
    print(f"R²  :  {r2:.4f}")
    print("\nApproximate dollar interpretation:")
    print(f"MAE :  ${mae * 100_000:,.0f}")
    print(f"RMSE:  ${rmse * 100_000:,.0f}")

    return predictions, {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_plots(train_losses, test_losses, actual, predictions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curves
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training and test loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=150)
    plt.close()

    # Actual vs predicted values
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predictions, alpha=0.25, s=10)
    lower = min(float(actual.min()), float(predictions.min()))
    upper = max(float(actual.max()), float(predictions.max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual house value ($100,000 units)")
    plt.ylabel("Predicted house value ($100,000 units)")
    plt.title("Actual vs. predicted house values")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


# -----------------------------
# 6. Main execution
# -----------------------------
def main():
    set_seed()

    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {', '.join(feature_names)}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = HousingRegressor(input_size=X_train.shape[1])
    print("\nModel architecture:")
    print(model)

    train_losses, test_losses = train_model(
        model,
        train_loader,
        X_test_tensor,
        y_test_tensor,
    )

    predictions, _ = evaluate_model(model, X_test_tensor, y_test)
    save_plots(
        train_losses,
        test_losses,
        y_test.reshape(-1),
        predictions,
    )

    print(f"\nPlots saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
