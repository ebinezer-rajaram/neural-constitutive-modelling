"""
Problem 1 — Material A: Neural Constitutive Model (Plane Strain)
Cambridge IIB Engineering — Module 4C11, Coursework 1

Trains a feedforward ResNet to learn the constitutive law for Material A.
Material A is a plane-strain material:
  - ε₃₃ (index 2) = 0 — no out-of-plane strain
  - σ₃₃ (index 2) ≠ 0 — non-zero through-thickness stress reaction
Data: 1100 samples × 6 components × 50 time steps

Network maps: flattened strain path [nstep×ndim] → flattened stress path [nstep×ndim]
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
EPOCHS       = 500   # Maximum epochs (early stopping will typically halt before this)
PATIENCE     = 50    # Early stopping: halt if test loss does not improve for this many epochs
BATCH_SIZE   = 20    # Small batches improve generalisation via gradient noise
LR           = 1e-3  # Adam default learning rate; works well for regression tasks
WEIGHT_DECAY = 1e-4  # L2 regularisation weight; reduces overfitting with minimal effect on convergence
HIDDEN_SIZE  = 256   # Hidden layer width — large enough for 6-component multiaxial problem
N_BLOCKS     = 4     # Number of residual blocks; depth captures nonlinear path-dependence
NTRAIN       = 880   # 80% of 1100 samples for training
NTEST        = 220   # 20% of 1100 samples for testing
SEED         = 42    # Random seed for reproducibility

torch.manual_seed(SEED)
np.random.seed(SEED)


class MatRead:
    """
    Reads MATLAB v7.3 (.mat) files, which use HDF5 format (requires h5py).
    scipy.io.loadmat cannot handle MATLAB v7.3 format.
    """
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')

    def get_strain(self):
        # MATLAB stores arrays in column-major (Fortran) order.
        # h5py reads HDF5 with dimensions reversed relative to MATLAB indexing.
        # Original MATLAB shape: (nsamples=1100, ncomponents=6, ntimesteps=50)
        # h5py reads as:         (ntimesteps=50, ncomponents=6, nsamples=1100)
        # After transpose(2,0,1): (nsamples=1100, ntimesteps=50, ncomponents=6)
        strain = np.array(self.data['strain']).transpose(2, 0, 1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2, 0, 1)
        return torch.tensor(stress, dtype=torch.float32)


class DataNormalizer:
    """
    Z-score normaliser fitted on training data only (prevents data leakage).
    Applied feature-wise: each of the nstep×ndim flattened features is
    independently shifted to mean=0 and scaled to std=1.
    """
    def __init__(self, data):
        # data shape: [n_samples, nstep*ndim] — already flattened
        self.mean = data.mean(dim=0, keepdim=True)   # shape [1, nstep*ndim]
        self.std  = data.std(dim=0, keepdim=True)
        # Clamp std to avoid division by zero for constant features (e.g. ε₃₃=0)
        self.std  = torch.clamp(self.std, min=1e-8)

    def encode(self, data):
        """Normalise: x → (x − μ) / σ"""
        return (data - self.mean) / self.std

    def decode(self, data):
        """Denormalise: x̂ → x̂ × σ + μ"""
        return data * self.std + self.mean


class ResBlock(nn.Module):
    """
    Residual block: two Linear layers with a skip connection.
    Skip connections help gradient flow through deep networks and allow the
    network to learn residual corrections rather than the full mapping.
    """
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        residual = x                    # Save input for skip connection
        x = F.relu(self.fc1(x))        # First linear + activation
        x = self.fc2(x)                # Second linear (no activation before skip add)
        return F.relu(x + residual)     # Add skip connection, then activate


class Const_Net(nn.Module):
    """
    ResNet constitutive model.

    Architecture:
        Linear(in_size → hidden) + ReLU        ← input projection
        × N_BLOCKS: ResBlock(hidden)            ← residual feature extraction
        Linear(hidden → out_size)               ← output projection (no activation)

    Input:  flattened strain path, shape [batch, nstep × ndim]
    Output: flattened stress path, shape [batch, nstep × ndim]

    No final activation because this is a regression task (stress can be
    positive or negative, and unbounded).
    """
    def __init__(self, in_size, out_size, hidden_size=256, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Linear(in_size, hidden_size)
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_size) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.input_proj(x))  # Project to hidden dimension
        for block in self.res_blocks:
            x = block(x)               # Pass through residual blocks
        return self.output_proj(x)     # Project to output dimension


class Lossfunc:
    """
    Mean Squared Error loss, computed in normalised space.
    MSE is standard for regression; using normalised targets ensures all
    stress components contribute equally regardless of magnitude.
    """
    def __call__(self, output, target):
        return F.mse_loss(output, target)


# Data processing
path        = 'data/Material_A.mat'
data_reader = MatRead(path)
strain      = data_reader.get_strain()   # [1100, 50, 6]
stress      = data_reader.get_stress()   # [1100, 50, 6]

# Plane-strain enforcement
# Material A is loaded under plane-strain conditions.
# The out-of-plane normal strain ε₃₃ (component index 2) must be zero.
# We explicitly set it to enforce this constraint and prevent the normaliser
# from treating it as a meaningful input feature.
strain[:, :, 2] = 0.0   # ε₃₃ = 0  (plane-strain constraint)
# σ₃₃ (stress index 2) is non-zero and is still predicted as a network output.

nsamples, nstep, ndim = strain.shape    # 1100, 50, 6
in_size = out_size = nstep * ndim       # 300

# Flatten time steps and components into a single feature vector per sample
strain_flat = strain.reshape(nsamples, -1)   # [1100, 300]
stress_flat = stress.reshape(nsamples, -1)   # [1100, 300]

# Train / test split (sequential; data is assumed to be randomly ordered)
train_strain = strain_flat[:NTRAIN]     # [880, 300]
test_strain  = strain_flat[NTRAIN:]     # [220, 300]
train_stress = stress_flat[:NTRAIN]
test_stress  = stress_flat[NTRAIN:]

# Normalise — fit normalisers on training data only
strain_normalizer   = DataNormalizer(train_strain)
train_strain_encode = strain_normalizer.encode(train_strain)
test_strain_encode  = strain_normalizer.encode(test_strain)

stress_normalizer   = DataNormalizer(train_stress)
train_stress_encode = stress_normalizer.encode(train_stress)
test_stress_encode  = stress_normalizer.encode(test_stress)

# DataLoader for mini-batch training
train_set    = Data.TensorDataset(train_strain_encode, train_stress_encode)
train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


# Model, optimiser, scheduler
net       = Const_Net(in_size, out_size, hidden_size=HIDDEN_SIZE, n_blocks=N_BLOCKS)
loss_func = Lossfunc()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# CosineAnnealingLR gradually reduces the learning rate from LR to near-zero
# over T_max epochs, preventing oscillation near convergence.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Number of parameters: {n_params}')


# Training loop
print(f'Start training for up to {EPOCHS} epochs (patience={PATIENCE})...')

loss_train_list = []
loss_test_list  = []

# Early stopping state
best_test_loss = float('inf')
best_epoch     = 0
best_state     = None   # deep copy of net.state_dict() at best epoch

for epoch in range(EPOCHS):
    net.train(True)
    trainloss = 0.0

    for strain_batch, stress_batch in train_loader:
        output = net(strain_batch)                  # Forward pass
        loss   = loss_func(output, stress_batch)    # MSE in normalised space

        optimizer.zero_grad()   # Clear accumulated gradients
        loss.backward()         # Backpropagate
        optimizer.step()        # Update network weights

        trainloss += loss.item()

    scheduler.step()            # Decay learning rate after each epoch

    # Evaluate on full test set (no gradient tracking needed)
    net.eval()
    with torch.no_grad():
        test_pred = net(test_strain_encode)
        testloss  = loss_func(test_pred, test_stress_encode).item()

    if epoch % 10 == 0:
        print(f'epoch: {epoch:4d},  '
              f'train loss: {trainloss / len(train_loader):.6f},  '
              f'test loss: {testloss:.6f}')

    loss_train_list.append(trainloss / len(train_loader))
    loss_test_list.append(testloss)

    # Early stopping: save best weights and break if no improvement for PATIENCE epochs
    if testloss < best_test_loss:
        best_test_loss = testloss
        best_epoch     = epoch
        best_state     = {k: v.clone() for k, v in net.state_dict().items()}

    if epoch - best_epoch >= PATIENCE:
        print(f'Early stopping at epoch {epoch}  '
              f'(best epoch {best_epoch}, best test loss {best_test_loss:.6f})')
        break

# Restore weights from the best epoch
net.load_state_dict(best_state)
print(f'Restored best model from epoch {best_epoch}')
print(f'Best train loss: {loss_train_list[best_epoch]:.6f}')
print(f'Best test  loss: {best_test_loss:.6f}')


# Plotting
os.makedirs('outputs', exist_ok=True)

COMPONENT_LABELS = ['σ₁₁', 'σ₂₂', 'σ₃₃', 'σ₁₂', 'σ₂₃', 'σ₁₃']

# Figure 1: Loss curves
plt.figure(1, figsize=(8, 5))
plt.semilogy(loss_train_list, label='Train loss', color='steelblue')
plt.semilogy(loss_test_list,  label='Test loss',  color='tomato', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Material A — Training and Test Loss vs Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/Material_A_loss.png', dpi=150)
print('Saved: outputs/Material_A_loss.png')

# Figure 2: Ground truth vs predicted stress — representative samples
# Compute per-sample RMSE in physical (Pa) space over all 220 test samples
# to select principled comparison examples instead of an arbitrary index.
net.eval()
with torch.no_grad():
    all_pred_enc = net(test_strain_encode)   # [220, 300]

all_pred_phys = stress_normalizer.decode(all_pred_enc)          # [220, 300]
all_true_phys = stress_normalizer.decode(test_stress_encode)    # [220, 300]

per_sample_rmse = torch.sqrt(((all_pred_phys - all_true_phys) ** 2).mean(dim=1))  # [220]

# Median-error sample: representative of typical model performance
# Worst-error sample: most diagnostic of model limitations
median_idx = int(torch.argsort(per_sample_rmse)[len(per_sample_rmse) // 2].item())
worst_idx  = int(torch.argmax(per_sample_rmse).item())
print(f'Median-error sample: idx={median_idx},  RMSE={per_sample_rmse[median_idx]:.4f} Pa')
print(f'Worst-error  sample: idx={worst_idx},   RMSE={per_sample_rmse[worst_idx]:.4f} Pa')

timesteps = np.arange(nstep)

for tag, sample_idx in [('median', median_idx), ('worst', worst_idx)]:
    pred_stress = all_pred_phys[sample_idx].reshape(nstep, ndim).numpy()
    true_stress = all_true_phys[sample_idx].reshape(nstep, ndim).numpy()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for i in range(ndim):
        rmse_i = np.sqrt(np.mean((pred_stress[:, i] - true_stress[:, i]) ** 2))
        axes[i].plot(timesteps, true_stress[:, i], 'b-',  lw=2,   label='Ground truth')
        axes[i].plot(timesteps, pred_stress[:, i], 'r--', lw=1.5, label='Predicted')
        axes[i].set_title(f'{COMPONENT_LABELS[i]}   RMSE={rmse_i:.3g} Pa')
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('Stress (Pa)')
        axes[i].legend(fontsize=10)

    rmse_total = per_sample_rmse[sample_idx].item()
    plt.suptitle(
        f'Material A — Ground-truth vs Predicted Stress '
        f'({tag}-error test sample, idx={sample_idx}, RMSE={rmse_total:.3g} Pa)',
        fontsize=11
    )
    plt.tight_layout()
    fname = f'outputs/Material_A_stress_{tag}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'Saved: {fname}')

# Figure 3: Distribution of per-sample RMSE across all test samples
plt.figure(figsize=(8, 4))
plt.hist(per_sample_rmse.numpy(), bins=30, color='steelblue', edgecolor='white')
plt.axvline(per_sample_rmse[median_idx].item(), color='orange', ls='--', lw=1.5,
            label=f'Median sample (idx={median_idx})')
plt.axvline(per_sample_rmse[worst_idx].item(),  color='red',    ls='--', lw=1.5,
            label=f'Worst sample  (idx={worst_idx})')
plt.xlabel('Per-sample RMSE (Pa)')
plt.ylabel('Count')
plt.title(f'Material A — Distribution of Test RMSE across {NTEST} test samples')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/Material_A_error_dist.png', dpi=150)
plt.close()
print('Saved: outputs/Material_A_error_dist.png')
