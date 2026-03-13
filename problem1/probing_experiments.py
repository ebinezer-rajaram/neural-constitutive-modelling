"""
Probing Experiments -- Problem 1, Materials A, B, C
Cambridge IIB Engineering 4C11 Coursework 1

Trains each neural constitutive model (same hyperparameters as individual scripts)
and probes the trained network with controlled, synthetic loading paths to identify
material behaviour:

  Elastic:        no hysteresis; loading and unloading stress-strain curves overlap
  Elastoplastic:  open hysteresis loops; residual stress when strain returns to zero
  Viscoelastic:   cannot be distinguished here -- time is implicit (index), not physical

Probing paths:
  Material A (plane-strain, 6 comp):
    Probe 1: Uniaxial load-unload          -- e11 triangle, others zero
    Probe 2: Pure shear load-unload        -- e12 triangle, others zero
    Probe 3: Equal biaxial load-unload     -- e11=e22 triangle, shear zero

  Material B (full 3D, 6 comp):
    Probe 1: Uniaxial tension-compression  -- e11 full sine cycle
    Probe 2: Shear cycle                   -- e12 full sine cycle
    Probe 3: Hydrostatic load-unload       -- e11=e22=e33 triangle

  Material C (uniaxial 1D):
    (Training data has e11 in [-0.071, 0] -- compressive only; probes kept in this range)
    Probe 1: Single compressive load-unload   -- e11 triangle to -0.05
    Probe 2: Two compressive cycles           -- repeated triangle
    Probe 3: Ratcheting path                  -- partial unloads, drifting mean compression

Figures saved to outputs/Material_{A,B,C}_probing.png
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

# ---- Hyperparameters (must match individual training scripts) ----------------
EPOCHS     = 500
BATCH_SIZE = 20
NTRAIN     = 880
SEED       = 42
NSTEP      = 50
os.makedirs('outputs', exist_ok=True)


# ---- Common classes (identical to individual training scripts) ---------------

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
        self.input_proj = nn.Linear(in_size, hidden_size)
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_size) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        for b in self.res_blocks:
            x = b(x)
        return self.output_proj(x)


# ---- Training helper ---------------------------------------------------------

def train_network(strain, stress, hidden_size, n_blocks, lr, label):
    """
    Train network and return (net, strain_normalizer, stress_normalizer).
    strain, stress: torch tensors [nsamples, nstep, ndim]
    """
    torch.manual_seed(SEED)
    nsamples, nstep, ndim = strain.shape
    io = nstep * ndim

    sf = strain.reshape(nsamples, -1)
    yf = stress.reshape(nsamples, -1)

    sn = DataNormalizer(sf[:NTRAIN])
    yn = DataNormalizer(yf[:NTRAIN])

    tx,  ty  = sn.encode(sf[:NTRAIN]), yn.encode(yf[:NTRAIN])
    tsx, tsy = sn.encode(sf[NTRAIN:]), yn.encode(yf[NTRAIN:])

    net    = Const_Net(io, io, hidden_size, n_blocks)
    opt    = torch.optim.Adam(net.parameters(), lr=lr)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = Data.DataLoader(Data.TensorDataset(tx, ty), BATCH_SIZE, shuffle=True)

    print(f'\nTraining {label} ({EPOCHS} epochs)...')
    for epoch in range(EPOCHS):
        net.train()
        for xb, yb in loader:
            loss = F.mse_loss(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        if (epoch + 1) % 100 == 0:
            net.eval()
            with torch.no_grad():
                tl = F.mse_loss(net(tsx), tsy).item()
            print(f'  epoch {epoch + 1:3d}: test loss = {tl:.5f}')

    net.eval()
    return net, sn, yn


# ---- Probe helper ------------------------------------------------------------

def predict_stress(net, sn, yn, strain_path):
    """
    Predict stress for one controlled strain path.
    strain_path: numpy array [nstep, ndim]
    Returns:     numpy array [nstep, ndim] of predicted stress
    """
    x = torch.tensor(strain_path.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        y = yn.decode(net(sn.encode(x)))
    return y.numpy().reshape(strain_path.shape)


# ---- Path generators --------------------------------------------------------

def triangle(amp, n=NSTEP):
    """Triangular load-unload: 0 -> amp -> 0 over n steps."""
    t = np.linspace(0.0, 1.0, n)
    return amp * (1.0 - 2.0 * np.abs(t - 0.5))


def two_cycles(amp, n=NSTEP):
    """Two triangular cycles: 0 -> amp -> 0 -> amp -> 0."""
    t = np.linspace(0.0, 2.0, n)
    return amp * (1.0 - 2.0 * np.abs((t % 1.0) - 0.5))


def sine_cycle(amp, n=NSTEP):
    """Full sinusoidal cycle: 0 -> +amp -> 0 -> -amp -> 0."""
    t = np.linspace(0.0, 1.0, n)
    return amp * np.sin(2.0 * np.pi * t)


def piecewise_path(keypoints, n=NSTEP):
    """Piecewise-linear path through keypoints, resampled to n steps."""
    tk = np.linspace(0.0, 1.0, len(keypoints))
    ta = np.linspace(0.0, 1.0, n)
    return np.interp(ta, tk, keypoints)


def loading_unloading_split(eps_1d):
    """Return (loading_slice, unloading_slice) split at the extremum of eps_1d."""
    pk = int(np.argmax(np.abs(eps_1d)))
    return slice(None, pk + 1), slice(pk, None)


# =============================================================================
# MATERIAL A -- Plane strain (e33 = e23 = e13 = 0 enforced)
# =============================================================================
print('\n' + '=' * 62)
print('MATERIAL A -- plane strain')
print('=' * 62)

reader_A  = MatRead('data/Material_A.mat')
strain_A  = reader_A.get_strain()
stress_A  = reader_A.get_stress()
strain_A[:, :, 2] = 0.0   # enforce plane-strain constraint

net_A, sn_A, yn_A = train_network(strain_A, stress_A, 256, 4, 1e-3, 'Material A')

AMP_A = 0.08   # safely within training range (~0.15 max)

# Build strain paths: shape [50, 6], components: [e11, e22, e33, e12, e23, e13]
def make_path_A(comp_amp):
    """comp_amp: dict {component_index: amplitude}"""
    path = np.zeros((NSTEP, 6))
    for c, amp in comp_amp.items():
        path[:, c] = triangle(amp)
    return path

pA1 = make_path_A({0: AMP_A})              # Uniaxial e11
pA2 = make_path_A({3: AMP_A})              # Pure shear e12
pA3 = make_path_A({0: AMP_A, 1: AMP_A})   # Equal biaxial e11 = e22

sA1 = predict_stress(net_A, sn_A, yn_A, pA1)
sA2 = predict_stress(net_A, sn_A, yn_A, pA2)
sA3 = predict_stress(net_A, sn_A, yn_A, pA3)

# -- Figure: Material A probing ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Probe 1: s11 vs e11 (uniaxial)
ld, un = loading_unloading_split(pA1[:, 0])
axes[0].plot(pA1[ld, 0], sA1[ld, 0], 'b-',  lw=2, label='Loading')
axes[0].plot(pA1[un, 0], sA1[un, 0], 'r--', lw=2, label='Unloading')
axes[0].set_xlabel('Strain e11')
axes[0].set_ylabel('Stress (Pa)')
axes[0].set_title('Probe 1: Uniaxial load-unload\ns11 vs e11')
axes[0].legend()

# Probe 2: s12 vs e12 (shear)
ld, un = loading_unloading_split(pA2[:, 3])
axes[1].plot(pA2[ld, 3], sA2[ld, 3], 'b-',  lw=2, label='Loading')
axes[1].plot(pA2[un, 3], sA2[un, 3], 'r--', lw=2, label='Unloading')
axes[1].set_xlabel('Strain e12')
axes[1].set_ylabel('Stress (Pa)')
axes[1].set_title('Probe 2: Pure shear load-unload\ns12 vs e12')
axes[1].legend()

# Probe 3: s11, s22 vs e11=e22 (biaxial) -- also show s33 (reaction stress)
ld, un = loading_unloading_split(pA3[:, 0])
for comp, name, col in [(0, 's11', 'b'), (1, 's22', 'r'), (2, 's33', 'g')]:
    axes[2].plot(pA3[ld, 0], sA3[ld, comp],  col + '-',  lw=2, label=f'{name}')
    axes[2].plot(pA3[un, 0], sA3[un, comp],  col + '--', lw=2)
axes[2].set_xlabel('Strain e11 = e22')
axes[2].set_ylabel('Stress (Pa)')
axes[2].set_title('Probe 3: Equal biaxial load-unload\ns11, s22, s33 vs e11=e22')
axes[2].legend(fontsize=8)
axes[2].text(0.05, 0.92,
             'Solid=loading  Dashed=unloading',
             transform=axes[2].transAxes, fontsize=7, color='grey')

plt.suptitle(
    'Material A -- Probing Experiments\n'
    'Elastic: loading and unloading curves overlap (no hysteresis)',
    fontsize=11
)
plt.tight_layout()
plt.savefig('outputs/Material_A_probing.png', dpi=150)
plt.close()
print('Saved: outputs/Material_A_probing.png')


# =============================================================================
# MATERIAL B -- Full 3D
# =============================================================================
print('\n' + '=' * 62)
print('MATERIAL B -- full 3D')
print('=' * 62)

reader_B  = MatRead('data/Material_B.mat')
net_B, sn_B, yn_B = train_network(
    reader_B.get_strain(), reader_B.get_stress(), 256, 4, 1e-3, 'Material B'
)

AMP_B = 0.07   # within training range

# Probe 1: uniaxial tension-compression sine cycle (e11 only)
pB1 = np.zeros((NSTEP, 6))
pB1[:, 0] = sine_cycle(AMP_B)

# Probe 2: shear sine cycle (e12 only)
pB2 = np.zeros((NSTEP, 6))
pB2[:, 3] = sine_cycle(AMP_B)

# Probe 3: hydrostatic load-unload (e11=e22=e33 triangle, shears zero)
pB3 = np.zeros((NSTEP, 6))
for c in [0, 1, 2]:
    pB3[:, c] = triangle(0.05)

sB1 = predict_stress(net_B, sn_B, yn_B, pB1)
sB2 = predict_stress(net_B, sn_B, yn_B, pB2)
sB3 = predict_stress(net_B, sn_B, yn_B, pB3)

# -- Figure: Material B probing -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Probe 1: s11 vs e11 -- uniaxial sine (shows hysteresis loop shape)
axes[0].plot(pB1[:, 0], sB1[:, 0], 'b-', lw=2)
axes[0].axhline(0, color='k', lw=0.5, ls=':')
axes[0].axvline(0, color='k', lw=0.5, ls=':')
# Direction arrow at first-quarter peak
q = NSTEP // 4
axes[0].annotate('', xy=(pB1[q + 1, 0], sB1[q + 1, 0]),
                 xytext=(pB1[q, 0], sB1[q, 0]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
axes[0].set_xlabel('Strain e11')
axes[0].set_ylabel('Stress (Pa)')
axes[0].set_title('Probe 1: Uniaxial tension-compression\ns11 vs e11  (arrow = direction)')

# Probe 2: s12 vs e12 -- shear sine
axes[1].plot(pB2[:, 3], sB2[:, 3], 'b-', lw=2)
axes[1].axhline(0, color='k', lw=0.5, ls=':')
axes[1].axvline(0, color='k', lw=0.5, ls=':')
q = NSTEP // 4
axes[1].annotate('', xy=(pB2[q + 1, 3], sB2[q + 1, 3]),
                 xytext=(pB2[q, 3], sB2[q, 3]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
axes[1].set_xlabel('Strain e12')
axes[1].set_ylabel('Stress (Pa)')
axes[1].set_title('Probe 2: Shear cycle\ns12 vs e12  (arrow = direction)')

# Probe 3: hydrostatic -- s11, s22, s33 vs e_vol (loading solid, unloading dashed)
ld, un = loading_unloading_split(pB3[:, 0])
for comp, name, col in [(0, 's11', 'b'), (1, 's22', 'r'), (2, 's33', 'g')]:
    axes[2].plot(pB3[ld, 0], sB3[ld, comp], col + '-',  lw=2, label=name)
    axes[2].plot(pB3[un, 0], sB3[un, comp], col + '--', lw=2)
axes[2].set_xlabel('Strain e11 = e22 = e33')
axes[2].set_ylabel('Stress (Pa)')
axes[2].set_title('Probe 3: Hydrostatic load-unload\ns11, s22, s33 vs e_hyd')
axes[2].legend(fontsize=8)
axes[2].text(0.05, 0.92,
             'Solid=loading  Dashed=unloading',
             transform=axes[2].transAxes, fontsize=7, color='grey')

plt.suptitle(
    'Material B -- Probing Experiments\n'
    'Elastic: loop closes; Inelastic: open hysteresis loop',
    fontsize=11
)
plt.tight_layout()
plt.savefig('outputs/Material_B_probing.png', dpi=150)
plt.close()
print('Saved: outputs/Material_B_probing.png')


# =============================================================================
# MATERIAL C -- Uniaxial 1D
# Training data: e11 in [-0.138, +0.141] -- both compressive and tensile.
# (Test sample 0 happened to be compressive-only; the full training set is not.)
# Probing paths use compressive-only paths (amplitude -0.05) as this is a
# natural loading regime for this material and keeps us well within distribution.
# =============================================================================
print('\n' + '=' * 62)
print('MATERIAL C -- uniaxial 1D')
print('=' * 62)

reader_C  = MatRead('data/Material_C.mat')
net_C, sn_C, yn_C = train_network(
    reader_C.get_strain(), reader_C.get_stress(), 128, 3, 1e-3, 'Material C'
)

AMP_C = -0.05   # compressive amplitude, within training range

# Probe 1: single compressive load-unload (0 -> -0.05 -> 0)
pC1 = triangle(AMP_C).reshape(NSTEP, 1)

# Probe 2: two compressive cycles (0 -> -0.05 -> 0 -> -0.05 -> 0)
pC2 = two_cycles(AMP_C).reshape(NSTEP, 1)

# Probe 3: ratcheting -- increasing mean compression with partial unloads
# 0 -> -0.03 -> -0.01 -> -0.05 -> -0.02 -> -0.06 -> -0.02 -> -0.06
pC3 = piecewise_path(
    [0.0, -0.03, -0.01, -0.05, -0.02, -0.06, -0.02, -0.06]
).reshape(NSTEP, 1)

sC1 = predict_stress(net_C, sn_C, yn_C, pC1)
sC2 = predict_stress(net_C, sn_C, yn_C, pC2)
sC3 = predict_stress(net_C, sn_C, yn_C, pC3)

# -- Figure: Material C probing -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Probe 1: load-unload -- do loading and unloading curves overlap?
ld, un = loading_unloading_split(pC1[:, 0])
axes[0].plot(pC1[ld, 0], sC1[ld, 0], 'b-',  lw=2.5, label='Loading   (0 to -0.05)')
axes[0].plot(pC1[un, 0], sC1[un, 0], 'r--', lw=2.5, label='Unloading (-0.05 to 0)')
axes[0].axhline(0, color='k', lw=0.5, ls=':')
axes[0].set_xlabel('Strain e11')
axes[0].set_ylabel('Stress s11 (Pa)')
axes[0].set_title('Probe 1: Compressive load-unload\nHysteresis gap = inelastic behaviour')
axes[0].legend(fontsize=8)

# Probe 2: two cycles -- do loops shift between cycles? (cyclic hardening/softening)
half = NSTEP // 2
axes[1].plot(pC2[:half, 0], sC2[:half, 0], 'b-',  lw=2.5, label='Cycle 1')
axes[1].plot(pC2[half:, 0], sC2[half:, 0], 'r--', lw=2.5, label='Cycle 2')
axes[1].axhline(0, color='k', lw=0.5, ls=':')
axes[1].set_xlabel('Strain e11')
axes[1].set_ylabel('Stress s11 (Pa)')
axes[1].set_title('Probe 2: Two compressive cycles\nLoop shift = cyclic hardening/softening')
axes[1].legend(fontsize=8)

# Probe 3: ratcheting -- does mean stress drift under repeated partial cycles?
axes[2].plot(pC3[:, 0], sC3[:, 0], 'b-', lw=2)
axes[2].scatter(pC3[0, 0],  sC3[0, 0],  color='green', zorder=5,
                s=80, label='Start', marker='o')
axes[2].scatter(pC3[-1, 0], sC3[-1, 0], color='red',   zorder=5,
                s=80, label='End',   marker='s')
axes[2].axhline(0, color='k', lw=0.5, ls=':')
axes[2].set_xlabel('Strain e11')
axes[2].set_ylabel('Stress s11 (Pa)')
axes[2].set_title('Probe 3: Ratcheting path\n(partial unloads, drifting mean strain)')
axes[2].legend(fontsize=8)

plt.suptitle(
    'Material C -- Probing Experiments (compressive loading, in-distribution)\n'
    'Elastic: loops close; Plastic: open loops, residual stress at e=0',
    fontsize=11
)
plt.tight_layout()
plt.savefig('outputs/Material_C_probing.png', dpi=150)
plt.close()
print('Saved: outputs/Material_C_probing.png')

print('\n\nAll probing experiments complete.')
print('Figures: outputs/Material_A_probing.png')
print('         outputs/Material_B_probing.png')
print('         outputs/Material_C_probing.png')
