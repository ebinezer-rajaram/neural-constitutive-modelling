# Neural Constitutive Modelling

Physics-informed machine learning for data-driven constitutive modelling in solid mechanics.
Cambridge IIB Engineering — Module 4C11, Coursework 1.

## Overview

- **Problem 1** — Learn stress-strain constitutive laws from experimental material data using a ResNet/U-Net architecture.
- **Problem 2** — Apply the trained constitutive model to certify structural integrity of the Eiffel Tower frame under random loading.

## Repository Structure

```
neural-constitutive-modelling/
├── docs/
│   └── Coursework1.pdf         # Assignment specification
├── problem1/
│   ├── data/
│   │   ├── Material_A.mat      # Multiaxial data: 1100 samples × 6 components × 50 steps
│   │   ├── Material_B.mat      # Multiaxial data: 1100 samples × 6 components × 50 steps
│   │   ├── Material_C.mat      # Uniaxial data:   1100 samples × 1 component  × 50 steps
│   │   └── README.txt          # Data format description
│   ├── outputs/                # Saved plots and model checkpoints (gitignored)
│   └── nn_skeleton.py          # Skeleton training script (to be completed)
└── problem2/
    ├── data/
    │   └── Eiffel_data.mat     # FE simulation data for structural analysis
    ├── matlab/
    │   ├── Eiffel.m            # Nodal coordinates and element connectivity
    │   ├── GenData_Eiffel.m    # FE data generation script
    │   ├── GenBoundary.m       # Random loading boundary condition generator
    │   └── randfixedsum.m      # Helper: random numbers with fixed sum
    └── outputs/                # Saved plots and results (gitignored)
```

## Requirements

- Python 3.x with PyTorch, NumPy, Matplotlib, h5py
- MATLAB (for Problem 2 data generation scripts)
