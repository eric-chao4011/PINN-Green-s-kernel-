# PINN Poisson Green's Operator

Learn a reusable solution operator for the two-dimensional Poisson equation using a physics-informed neural approximation of its Green's kernel. Numerical quadrature maps different forcing functions to their corresponding solutions.

## Problem and method

- Domain: [-1, 1]², with Δu = f and homogeneous Dirichlet boundary conditions u = 0.
- Prediction: u(x,y) ≈ ∬ Gθ(x,y,s,t) f(s,t) ds dt.
- **Plan A (2D):** two independent networks encode (x,y) and (s,t). A rank-50 inner product produces a separable kernel approximation.
- **Plan B (4D):** a network takes all four coordinates and predicts a scalar kernel value directly.
- Both architectures use fixed random Fourier features, GELU activations, and Gauss–Lobatto–Legendre (GLL) quadrature.
- Training pairs come from random sine Fourier fields and their analytic Laplacians.
- The loss combines solution reconstruction mean squared error with a weighted discrete Poisson residual. The current weight is λ = 1e-4.
- Spatial derivatives use GLL spectral differentiation matrices. Ngrid = 22 gives 23 nodes per axis.

## Project files

| Path | Purpose |
| --- | --- |
| main2.ipynb | Network definitions, training, prediction, and plots |
| test.ipynb / test_functions.py | Fixed benchmark cases and reproducible evaluation |
| test_results.csv | Full-precision benchmark metrics |
| config.py | Network and training settings |
| data_generator2.py | Random Fourier training data |
| grid.py / diff.py | GLL nodes, quadrature weights, and differentiation matrices |
| Models_PDEloss_2D / Models_PDEloss_4D | Saved models, experiment settings, and result images |
| paper/paper.pdf | Research paper |
| report2/ | Presentation LaTeX source, figures, and PDF |

## Quick start

Download this repository using **Code > Download ZIP** and extract it, or clone it with Git. Open a terminal in the project folder containing this README.

Prerequisites: **64-bit Python 3.12** and an internet connection for the initial dependency installation. A GPU is not required for running the supplied model. The notebook has been run on Windows with Python 3.12.14 and PyTorch 2.11.0 using the CPU.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

macOS/Linux and GPU execution have not been verified for this release. For GPU training, install a PyTorch/torchvision build compatible with your CUDA environment. The code selects CUDA when available and otherwise uses the CPU.

Open **main2.ipynb**, select the Python 3 kernel from this virtual environment, and run the cells in order. Keep the working directory at the project root.

The default configuration is `TRAIN = False` with Plan B. The notebook loads the supplied model, predicts the solution defined in `config.py`, prints the maximum nodal error, and displays solution and error plots. Generated information files and images are saved in `outputs/evaluation/`. The included model directories are preserved.

The checkpoints contain full PyTorch model objects and are loaded with `weights_only=False`. Only load checkpoints you trust.

## Reproduced benchmark results

The fixed benchmark notebook evaluates both supplied checkpoints with the same GLL grid, analytical forcing functions, and error definitions. `sup norm` is the maximum nodal absolute error, while `relative L2` is `||u_pred - u||₂ / ||u||₂`.

| Test | Model | Sup norm | Relative L2 |
| --- | --- | ---: | ---: |
| Mixed frequency | Plan A (factorized 2D) | 8.80e-1 | 6.53e-1 |
| Mixed frequency | Plan B (direct 4D) | 1.03e+0 | 5.73e-1 |
| Polynomial | Plan A (factorized 2D) | 6.60e-2 | 4.79e-2 |
| Polynomial | Plan B (direct 4D) | 5.95e-2 | 4.70e-2 |

The mixed-frequency case illustrates why the two metrics should be reported separately: Plan B has a slightly larger worst-node error but a lower error over the full grid. The polynomial case is much more accurate for both models. Run `test.ipynb` to reproduce the values recorded in `test_results.csv`.

## Train a model

Set `TRAIN = True` in `config.py`, then restart the notebook kernel and run `main2.ipynb` in order. New training outputs are written to `outputs/training/`, which is excluded from Git by default.

- **Plan B:** use `[4, 50, 50, 1]` for the network dimensions in `model_case`.
- **Plan A:** use `[2, 50, 50, 50]`; the final value is the factorization rank.

In non-training mode, the notebook selects `Models_PDEloss_2D` or `Models_PDEloss_4D` from the input dimension of the first entry in `model_case`. Use one architecture at a time. To load a newly trained model, set `Folder` in the notebook to `outputs/training` after switching `TRAIN` to `False`.

The full default training run uses 10,001 optimizer steps and a batch size of 16. The four-dimensional tensors can require substantial RAM or GPU memory. For a first training experiment, reduce `Batch_Size` and `Adam_iter`; this changes the experiment and is not a reproduction of the original training run. The full training run has not been revalidated for this release.

After editing the configuration, restart the notebook kernel and rerun the cells from the beginning to avoid retaining previously imported settings.

## Notes

- The paper and benchmark table use the fixed test cases in `test.ipynb`; reported values are rounded from `test_results.csv`.
- The paper describes a training Fourier cutoff of K=4. The current generator uses `config.RANKS=5` for the first sample in each batch and a hard-coded cutoff of 4 for subsequent samples. This setting is separate from Plan A's factorization rank of 50.
- The factorized implementation still explicitly constructs the four-dimensional kernel tensor; it does not yet implement a fully memory-efficient factorized quadrature contraction.
- Stored notebook outputs have been cleared. Saved model files and result images are included separately.
- No open-source license has been selected.
