"""
Problem 2 — Eiffel Tower Structural Certification: ResNet-style Classifier
Cambridge IIB Engineering — Module 4C11, Coursework 1

Binary classification task: predict whether the Eiffel Tower truss structure
survives (output=1) or fails (output=0) under a given distributed pressure loading.

Failure criterion: max element stress exceeds 500 MPa (yield_stress = 5e8 Pa).

Network maps: load_apply [nSamples × 20] → survival probability [nSamples × 1]

Architecture: ResNet-style fully connected network with residual skip connections.
    Skip connections allow the network to learn residual corrections rather than
    the full mapping, improving gradient flow through deeper networks.

    Input(20) → Linear(256)
              → ResBlock(256) × 3   [each: Linear→BN→ReLU→Linear→BN + skip → ReLU]
              → Linear(128) + ReLU
              → Linear(1)           [raw logit]
"""

import matplotlib
matplotlib.use('Agg')          # Non-interactive backend; use 'TkAgg' for pop-up windows
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import h5py
import os

# Hyperparameters (all in one place for easy adjustment)
EPOCHS      = 200          # Training epochs
BATCH_SIZE  = 32           # Mini-batch size
LR          = 1e-3         # Adam learning rate
HIDDEN_DIM  = 256          # Width of residual blocks
N_BLOCKS    = 3            # Number of residual blocks
NTRAIN_FRAC = 0.8          # 80% training, 20% test
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class MatRead:
    """
    Reads MATLAB v7.3 (.mat) files using h5py (HDF5 format).
    scipy.io.loadmat cannot handle MATLAB v7.3 files.

    Expected keys:
        load_apply: [nSamples × 20] applied forces at 20 chosen nodes
        result:     [nSamples × 1]  binary label (1=survives, 0=fails)
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')

    def get_loads(self):
        # h5py reads HDF5 with dimensions transposed relative to MATLAB.
        # MATLAB shape: (nSamples, 20) → h5py reads as (20, nSamples)
        # Transpose to restore (nSamples, 20)
        raw = np.array(self.data['load_apply'])
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T
        return torch.tensor(raw, dtype=torch.float32)

    def get_labels(self):
        # MATLAB shape: (nSamples, 1) → h5py reads as (1, nSamples)
        raw = np.array(self.data['result'])
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T
        return torch.tensor(raw, dtype=torch.float32)


class DataNormalizer:
    """
    Z-score normaliser fitted on training data only (prevents data leakage).
    Each of the 20 load features is independently normalised to mean=0, std=1.
    """
    def __init__(self, data):
        self.mean = data.mean(dim=0, keepdim=True)   # shape [1, 20]
        self.std  = data.std(dim=0, keepdim=True)
        self.std  = torch.clamp(self.std, min=1e-8)

    def encode(self, data):
        """Normalise: x → (x − μ) / σ"""
        return (data - self.mean) / self.std

    def decode(self, data):
        """Denormalise: x̂ → x̂ × σ + μ"""
        return data * self.std + self.mean


class ResBlock(nn.Module):
    """
    Residual block for fully connected layers.

    Structure:
        Linear(size → size) → BatchNorm → ReLU
        Linear(size → size) → BatchNorm
        + skip connection (identity, since dimensions are equal)
        → ReLU

    BatchNorm stabilises training by reducing internal covariate shift.
    The skip connection adds the input directly to the output, allowing
    the block to learn corrections (residuals) to the identity mapping.
    """
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)

    def forward(self, x):
        residual = x                         # Save input for skip connection
        out = F.relu(self.bn1(self.fc1(x)))  # First linear + BN + ReLU
        out = self.bn2(self.fc2(out))        # Second linear + BN (no activation yet)
        return F.relu(out + residual)        # Add skip connection, then activate


class ResNet_Classifier(nn.Module):
    """
    ResNet-style binary classifier.

    Architecture:
        Linear(input_dim → hidden_dim) + ReLU    ← input projection
        × N_BLOCKS: ResBlock(hidden_dim)          ← residual feature extraction
        Linear(hidden_dim → 128) + ReLU          ← compression
        Linear(128 → 1)                           ← raw logit output

    Input:  normalised load profile, shape [batch, 20]
    Output: raw logit, shape [batch, 1]
    """
    def __init__(self, input_dim, hidden_dim, n_blocks):
        super().__init__()
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(n_blocks)]
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = F.relu(self.input_proj(x))   # Project to hidden dimension
        for block in self.res_blocks:
            x = block(x)                 # Pass through residual blocks
        return self.classifier(x)        # Final classification logit


# Data processing
data_path   = 'data/Eiffel_data.mat'
data_reader = MatRead(data_path)
loads       = data_reader.get_loads()    # [nSamples, 20]
labels      = data_reader.get_labels()  # [nSamples, 1]

nsamples  = loads.shape[0]
input_dim = loads.shape[1]
ntrain    = int(nsamples * NTRAIN_FRAC)
ntest     = nsamples - ntrain

print(f"\nDataset: {nsamples} samples, input_dim={input_dim}")
print(f"Train: {ntrain}, Test: {ntest}")

# Sequential split
train_loads  = loads[:ntrain]
test_loads   = loads[ntrain:]
train_labels = labels[:ntrain]
test_labels  = labels[ntrain:]

# Class imbalance check
n_pos = train_labels.sum().item()
n_neg = ntrain - n_pos
ratio = n_pos / max(n_neg, 1)
print(f"\nClass distribution (train): {int(n_pos)} survive, {int(n_neg)} fail — ratio {ratio:.3f}")

if ratio < 0.3 or ratio > 3.0:
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"Imbalanced dataset detected — using BCEWithLogitsLoss with pos_weight={pos_weight.item():.3f}")
else:
    pos_weight = None
    print("Balanced dataset — using BCEWithLogitsLoss without pos_weight")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Normalise inputs (fit on training data only)
load_normalizer  = DataNormalizer(train_loads)
train_loads_enc  = load_normalizer.encode(train_loads)
test_loads_enc   = load_normalizer.encode(test_loads)

# DataLoader for mini-batch training
train_set    = Data.TensorDataset(train_loads_enc, train_labels)
train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


# Model, optimiser, scheduler
net       = ResNet_Classifier(input_dim, HIDDEN_DIM, N_BLOCKS)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\nResNet Classifier — Number of trainable parameters: {n_params:,}")


# Training loop
print(f"\nStart training for {EPOCHS} epochs...")

loss_train_list = []
loss_test_list  = []

for epoch in range(EPOCHS):
    net.train(True)
    trainloss = 0.0

    for load_batch, label_batch in train_loader:
        logits = net(load_batch)                  # Forward pass → raw logit
        loss   = criterion(logits, label_batch)   # BCE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainloss += loss.item()

    scheduler.step()

    net.eval()
    with torch.no_grad():
        test_logits = net(test_loads_enc)
        testloss    = criterion(test_logits, test_labels).item()

    if epoch % 10 == 0:
        print(f'epoch: {epoch:4d},  '
              f'train loss: {trainloss / len(train_loader):.6f},  '
              f'test loss: {testloss:.6f}')

    loss_train_list.append(trainloss / len(train_loader))
    loss_test_list.append(testloss)

print(f'Final train loss: {loss_train_list[-1]:.6f}')
print(f'Final test  loss: {loss_test_list[-1]:.6f}')


# Evaluation
net.eval()
with torch.no_grad():
    test_logits = net(test_loads_enc)
    test_probs  = torch.sigmoid(test_logits)
    test_preds  = (test_probs >= 0.5).float()

tp = ((test_preds == 1) & (test_labels == 1)).sum().item()
tn = ((test_preds == 0) & (test_labels == 0)).sum().item()
fp = ((test_preds == 1) & (test_labels == 0)).sum().item()
fn = ((test_preds == 0) & (test_labels == 1)).sum().item()

accuracy  = (tp + tn) / ntest
precision = tp / max(tp + fp, 1)
recall    = tp / max(tp + fn, 1)

print(f"\nTest Accuracy:  {accuracy * 100:.2f}%")
print(f"Test Precision: {precision * 100:.2f}%")
print(f"Test Recall:    {recall * 100:.2f}%")


# Plotting
os.makedirs('outputs', exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(loss_train_list, label='Train loss', color='steelblue')
plt.plot(loss_test_list,  label='Test loss',  color='tomato', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('Problem 2 ResNet — Training and Test Loss vs Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/Problem2_ResNet_loss.png', dpi=150)
print('\nSaved: outputs/Problem2_ResNet_loss.png')


# Summary
print("\n" + "="*60)
print("SUMMARY — Problem 2: ResNet Classifier")
print("="*60)
print(f"Architecture : ResNet ({N_BLOCKS} residual blocks, hidden_dim={HIDDEN_DIM})")
print(f"Parameters   : {n_params:,}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision    : {precision * 100:.2f}%")
print(f"Recall       : {recall * 100:.2f}%")
print("="*60)
