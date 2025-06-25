import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import MSELoss
from torch.optim import Adam
from MPNN import MPNN
from DataLoader import train_loader, val_loader, test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MPNN().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = MSELoss()

def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).cpu()
            y = batch.y.cpu()
            y_pred.append(pred)
            y_true.append(y)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return y_true, y_pred

def calculate_mae_rmse(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

best_val_rmse = np.inf
for epoch in range(1, 191):
    loss = train_one_epoch()
    y_val, y_val_pred = evaluate(val_loader)
    mae_val, rmse_val = calculate_mae_rmse(y_val, y_val_pred)
    print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val MAE: {mae_val:.4f} | Val RMSE: {rmse_val:.4f}")
    if rmse_val < best_val_rmse:
        best_val_rmse = rmse_val
        torch.save(model.state_dict(), "best_gine_model.pt")

model.load_state_dict(torch.load("best_gine_model.pt"))
y_test, y_test_pred = evaluate(test_loader)
mae_test, rmse_test = calculate_mae_rmse(y_test, y_test_pred)
r2_test = r2_score_manual(y_test, y_test_pred)
print(f"Best Val RMSE: {best_val_rmse:.4f} | Test MAE: {mae_test:.4f} | Test RMSE: {rmse_test:.4f} | Test R2: {r2_test:.4f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"results/{timestamp}"
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "true_vs_pred.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["True_LMP_Spread", "Predicted_LMP_Spread"])
    for t, p in zip(y_test, y_test_pred):
        writer.writerow([f"{t:.6f}", f"{p:.6f}"])

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='k', linewidth=0.5)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
plt.xlabel("True LMP Spread")
plt.ylabel("Predicted LMP Spread")
plt.title("True vs Predicted LMP Spread")
plt.text(0.05, 0.95, f"$R^2$ = {r2_test:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(out_dir, "true_vs_pred.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved results to: {out_dir}")