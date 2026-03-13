"""
Problem 2 — Eiffel Tower Structural Certification: U-Net-style Classifier
Cambridge IIB Engineering — Module 4C11, Coursework 1

Binary classification task: predict whether the Eiffel Tower truss structure
survives (output=1) or fails (output=0) under a given distributed pressure loading.

Failure criterion: max element stress exceeds 500 MPa (yield_stress = 5e8 Pa).

Network maps: load_apply [nSamples × 20] → survival probability [nSamples × 1]

Architecture: U-Net-style encoder-decoder adapted to 1D fully connected layers.
    The encoder progressively compresses the input, creating a bottleneck
    representation. The decoder expands it back, with skip connections
    concatenating matching encoder features at each level. This allows the
    decoder to recover fine-grained information that would otherwise be lost
    during compression.

    Encoder:
        enc1: Linear(20  → 256) + BN + ReLU  → h1 [256]
        enc2: Linear(256 → 128) + BN + ReLU  → h2 [128]
        enc3: Linear(128 → 64)  + BN + ReLU  → h3 [64]
        enc4: Linear(64  → 32)  + BN + ReLU  → h4 [32]  [bottleneck]

    Decoder (skip = concat encoder output at matching level):
        dec4: Linear(32+64   → 64)  + BN + ReLU  [concat h4 + h3]
        dec3: Linear(64+128  → 128) + BN + ReLU  [concat dec4_out + h2]
        dec2: Linear(128+256 → 256) + BN + ReLU  [concat dec3_out + h1]

    Classifier head:
        Linear(256 → 128) + ReLU
        Linear(128 → 1)            [raw logit]
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


def enc_block(in_dim, out_dim):
    """Single encoder stage: Linear → BatchNorm → ReLU."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU()
    )


class UNet_Classifier(nn.Module):
    """
    U-Net-style binary classifier adapted to 1D fully connected layers.

    Encoder path compresses the input through four stages, saving feature
    maps (h1–h4) at each level. The decoder expands the bottleneck representation,
    concatenating the matching encoder feature map at each stage (skip connections).
    Skip connections preserve spatial/feature detail lost during compression and
    give the classifier access to both high-level (compressed) and low-level
    (detailed) representations simultaneously.

    Input:  normalised load profile, shape [batch, 20]
    Output: raw logit, shape [batch, 1]
    """
    def __init__(self, input_dim):
        super().__init__()

        # Encoder
        self.enc1 = enc_block(input_dim, 256)   # 20  → 256, saves h1
        self.enc2 = enc_block(256, 128)          # 256 → 128, saves h2
        self.enc3 = enc_block(128, 64)           # 128 → 64,  saves h3
        self.enc4 = enc_block(64, 32)            # 64  → 32   [bottleneck, h4]

        # Decoder
        # Each decoder stage concatenates the bottleneck/previous decoder output
        # with the matching encoder skip feature, doubling the input dimension.
        self.dec4 = enc_block(32 + 64, 64)      # concat(h4[32], h3[64]) → 64
        self.dec3 = enc_block(64 + 128, 128)    # concat(dec4[64], h2[128]) → 128
        self.dec2 = enc_block(128 + 256, 256)   # concat(dec3[128], h1[256]) → 256

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)    # Raw logit output
        )

    def forward(self, x):
        # Encoder — save intermediate features for skip connections
        h1 = self.enc1(x)    # [batch, 256]
        h2 = self.enc2(h1)   # [batch, 128]
        h3 = self.enc3(h2)   # [batch, 64]
        h4 = self.enc4(h3)   # [batch, 32]  ← bottleneck

        # Decoder — concatenate skip connections from encoder
        d4 = self.dec4(torch.cat([h4, h3], dim=1))   # concat(32+64) → 64
        d3 = self.dec3(torch.cat([d4, h2], dim=1))   # concat(64+128) → 128
        d2 = self.dec2(torch.cat([d3, h1], dim=1))   # concat(128+256) → 256

        return self.head(d2)   # Final classification logit


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
net       = UNet_Classifier(input_dim)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\nU-Net Classifier — Number of trainable parameters: {n_params:,}")


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
plt.title('Problem 2 U-Net — Training and Test Loss vs Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/Problem2_UNet_loss.png', dpi=150)
print('\nSaved: outputs/Problem2_UNet_loss.png')


# Summary
print("\n" + "="*60)
print("SUMMARY — Problem 2: U-Net Classifier")
print("="*60)
print(f"Architecture : U-Net-style (encoder-decoder with skip connections)")
print(f"Encoder dims : 20 → 256 → 128 → 64 → 32 (bottleneck)")
print(f"Decoder dims : 32+64→64 → 64+128→128 → 128+256→256")
print(f"Parameters   : {n_params:,}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision    : {precision * 100:.2f}%")
print(f"Recall       : {recall * 100:.2f}%")
print("="*60)
