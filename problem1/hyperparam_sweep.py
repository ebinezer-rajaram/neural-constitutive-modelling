"""
Hyperparameter sweep for Problem 1 — Materials A, B, C
Cambridge IIB Engineering — Module 4C11, Coursework 1

Trains each configuration for SWEEP_EPOCHS epochs and records final
train/test loss. Prints a comparison table and saves loss curve plots.

Run from the problem1/ directory:
    python hyperparam_sweep.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import h5py
import os

# ── Sweep settings ────────────────────────────────────────────────────────────
SWEEP_EPOCHS = 200   # Reduced from 500 so the sweep finishes quickly
BATCH_SIZE   = 20
NTRAIN       = 880
SEED         = 42
os.makedirs('outputs', exist_ok=True)

# ── Configurations to test ───────────────────────────────────────────────────
# Format: (label, hidden_size, n_blocks, lr, weight_decay)
CONFIGS_AB = [
    ('small  | h=128 b=2 lr=1e-3', 128, 2, 1e-3, 0.0),
    ('base   | h=256 b=4 lr=1e-3', 256, 4, 1e-3, 0.0),   # <-- our default
    ('large  | h=512 b=4 lr=1e-3', 512, 4, 1e-3, 0.0),
    ('hi-lr  | h=256 b=4 lr=1e-2', 256, 4, 1e-2, 0.0),
    ('lo-lr  | h=256 b=4 lr=1e-4', 256, 4, 1e-4, 0.0),
    ('wd1e-4 | h=256 b=4 lr=1e-3', 256, 4, 1e-3, 1e-4),  # L2 regularisation
    ('wd1e-3 | h=256 b=4 lr=1e-3', 256, 4, 1e-3, 1e-3),  # Stronger L2 regularisation
]

CONFIGS_C = [
    ('small  | h=64  b=2 lr=1e-3',  64, 2, 1e-3, 0.0),
    ('base   | h=128 b=3 lr=1e-3', 128, 3, 1e-3, 0.0),   # <-- our default
    ('large  | h=256 b=4 lr=1e-3', 256, 4, 1e-3, 0.0),
    ('hi-lr  | h=128 b=3 lr=1e-2', 128, 3, 1e-2, 0.0),
    ('lo-lr  | h=128 b=3 lr=1e-4', 128, 3, 1e-4, 0.0),
    ('wd1e-4 | h=128 b=3 lr=1e-3', 128, 3, 1e-3, 1e-4),  # L2 regularisation
    ('wd1e-3 | h=128 b=3 lr=1e-3', 128, 3, 1e-3, 1e-3),  # Stronger L2 regularisation
]


# ── Helper classes (same as individual scripts) ───────────────────────────────

class MatRead:
    def __init__(self, path):
        self.data = h5py.File(path, 'r')

    def get_strain(self):
        return torch.tensor(
            np.array(self.data['strain']).transpose(2, 0, 1), dtype=torch.float32
        )

    def get_stress(self):
        return torch.tensor(
            np.array(self.data['stress']).transpose(2, 0, 1), dtype=torch.float32
        )


class DataNormalizer:
    def __init__(self, data):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std  = torch.clamp(data.std(dim=0, keepdim=True), min=1e-8)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        r = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x + r)


class Const_Net(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, n_blocks):
        super().__init__()
        self.input_proj  = nn.Linear(in_size, hidden_size)
        self.res_blocks  = nn.ModuleList([ResBlock(hidden_size) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        for b in self.res_blocks:
            x = b(x)
        return self.output_proj(x)


# ── Core train/eval function ──────────────────────────────────────────────────

def run_config(train_x, train_y, test_x, test_y, in_size, out_size,
               hidden_size, n_blocks, lr, epochs, label, weight_decay=0.0):
    """Train one configuration, return (train_losses, test_losses, n_params)."""
    torch.manual_seed(SEED)
    net       = Const_Net(in_size, out_size, hidden_size, n_blocks)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_set    = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        net.train()
        tloss = 0.0
        for xb, yb in train_loader:
            pred = net(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            testloss = F.mse_loss(net(test_x), test_y).item()

        train_losses.append(tloss / len(train_loader))
        test_losses.append(testloss)

        if (epoch + 1) % 50 == 0:
            print(f'  [{label}]  epoch {epoch+1:3d}  '
                  f'train={train_losses[-1]:.5f}  test={testloss:.5f}')

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return train_losses, test_losses, n_params


# ── Per-material sweep ────────────────────────────────────────────────────────

def sweep_material(mat_name, data_path, configs, zero_eps33=False):
    print(f'\n{"="*65}')
    print(f'  Sweeping Material {mat_name}')
    print(f'{"="*65}')

    reader = MatRead(data_path)
    strain = reader.get_strain()
    stress = reader.get_stress()

    if zero_eps33:
        strain[:, :, 2] = 0.0   # plane-strain: ε₃₃ = 0

    nsamples, nstep, ndim = strain.shape
    io_size = nstep * ndim

    strain_flat = strain.reshape(nsamples, -1)
    stress_flat = stress.reshape(nsamples, -1)

    train_s   = strain_flat[:NTRAIN]
    test_s    = strain_flat[NTRAIN:]
    train_sig = stress_flat[:NTRAIN]
    test_sig  = stress_flat[NTRAIN:]

    sn = DataNormalizer(train_s);   tx = sn.encode(train_s);  tsx = sn.encode(test_s)
    yn = DataNormalizer(train_sig); ty = yn.encode(train_sig); tsy = yn.encode(test_sig)

    results = []
    all_train, all_test = [], []

    for label, hidden, blocks, lr, wd in configs:
        print(f'\n  Config: {label}')
        trl, tel, np_ = run_config(
            tx, ty, tsx, tsy, io_size, io_size,
            hidden, blocks, lr, SWEEP_EPOCHS, label, weight_decay=wd
        )
        results.append((label, hidden, blocks, lr, wd, np_, trl[-1], tel[-1]))
        all_train.append(trl)
        all_test.append(tel)

    # ── Print results table ───────────────────────────────────────────────────
    print(f'\n  Results — Material {mat_name} ({SWEEP_EPOCHS} epochs)')
    print(f'  {"Config":<35} {"Params":>8}  {"Train loss":>12}  {"Test loss":>12}')
    print(f'  {"-"*35} {"-"*8}  {"-"*12}  {"-"*12}')
    for label, h, b, lr, wd, np_, trl, tel in results:
        marker = '  <-- best test' if tel == min(r[7] for r in results) else ''
        print(f'  {label:<35} {np_:>8,}  {trl:>12.6f}  {tel:>12.6f}{marker}')

    # ── Loss curve comparison plot ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, SWEEP_EPOCHS + 1)

    for (label, *_), trl, tel in zip(configs, all_train, all_test):
        ax1.semilogy(epochs_range, trl, label=label)
        ax2.semilogy(epochs_range, tel, label=label)

    ax1.set_title(f'Material {mat_name} — Train Loss')
    ax2.set_title(f'Material {mat_name} — Test Loss')
    for ax in (ax1, ax2):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (log scale)')
        ax.legend(fontsize=7)

    plt.suptitle(f'Material {mat_name} — Hyperparameter Sweep ({SWEEP_EPOCHS} epochs)')
    plt.tight_layout()
    fname = f'outputs/Material_{mat_name}_sweep.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'\n  Saved: {fname}')

    return results


# ── Run all three materials ───────────────────────────────────────────────────

results_A = sweep_material('A', 'data/Material_A.mat', CONFIGS_AB, zero_eps33=True)
results_B = sweep_material('B', 'data/Material_B.mat', CONFIGS_AB, zero_eps33=False)
results_C = sweep_material('C', 'data/Material_C.mat', CONFIGS_C,  zero_eps33=False)

print('\nDone. Sweep plots saved to outputs/')
