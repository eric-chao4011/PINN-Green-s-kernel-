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
| evaluate.py | Command-line evaluation and optional reference comparison |
| main2.ipynb | Network definitions, training, evaluation, and plots |
| config.py | Network and training settings |
| data_generator2.py | Random Fourier training data |
| grid.py / diff.py | GLL nodes, quadrature weights, and differentiation matrices |
| test.ipynb / test_functions.py | Generalization tests for the two saved models |
| test_results.csv | Saved evaluation results |
| Models_PDEloss_2D / Models_PDEloss_4D | Saved models, experiment settings, and result images |
| paper/ | Paper LaTeX source, figures, and PDF |
| report2/ | Presentation LaTeX source, figures, and PDF |

## Quick start

Download this repository using **Code > Download ZIP** and extract it, or clone it with Git. Open a terminal in the extracted project folder (the folder containing this README).

Prerequisites: **64-bit Python 3.12** and an internet connection for the initial dependency installation. A GPU is not required for evaluating the supplied models. Python 3.13 was used in the original experiments, and this release has been checked with Python 3.12.14 on Windows using CPU execution.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluate.py --check
```

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python evaluate.py --check
```

The macOS/Linux commands use standard virtual-environment paths; these platforms have not been tested for this release. For GPU training, install a PyTorch/torchvision build compatible with your CUDA environment. The code selects CUDA when available and otherwise uses the CPU.

### Expected output

`evaluate.py` executes `test.ipynb` in a fresh kernel, prints the four evaluation rows, and writes:

- `outputs/test_results.csv`: newly computed errors.
- `outputs/evaluation.ipynb`: the executed notebook.

With `--check`, it also compares all four cases against the reference `test_results.csv` using relative tolerance 1e-5 and absolute tolerance 1e-8. Successful reproduction ends with:

```text
PASS: all four cases match the reference (rtol=1e-5, atol=1e-8).
```

The reference CSV and included models are preserved. Generated files and the virtual environment are excluded by `.gitignore`.

### Interactive notebooks

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

On macOS/Linux, use `.venv/bin/python -m jupyterlab`. Select the Python 3 kernel from this virtual environment, open `test.ipynb`, and run the cells in order. Keep the notebook working directory at the project root.

The checkpoints contain full PyTorch model objects and are loaded with `weights_only=False`. Only load checkpoints you trust.

## Train a model

Set `TRAIN = True` in `config.py`, then run `main2.ipynb` in order. New training outputs are written to `outputs/training/`, which is excluded from Git by default.

- **Plan B:** use `[4, 50, 50, 1]` for the network dimensions in `model_case`.
- **Plan A:** use `[2, 50, 50, 50]`; the final value is the factorization rank.

The default configuration is `TRAIN = False` with Plan B. In evaluation mode, `main2.ipynb` selects `Models_PDEloss_2D` or `Models_PDEloss_4D` from the input dimension of the first entry in `model_case`. Use one architecture at a time; additional or mixed experiments require matching output paths. Evaluation information files and plots are written to `outputs/evaluation/`; the included model directories are preserved.

The full default training run uses 10,001 optimizer steps and a batch size of 16. The four-dimensional tensors can require substantial RAM or GPU memory. For a first training experiment, reduce `Batch_Size` and `Adam_iter`; this changes the experiment and is not a reproduction of the original training run.

After editing the configuration, restart the notebook kernel and rerun the cells from the beginning to avoid retaining previously imported settings.

## Saved results

These values come from the reference `test_results.csv`. A fresh Windows/Python 3.12 CPU evaluation reproduced all four rows within the tolerances used by `evaluate.py --check`. `relative_L2` is the unweighted relative Euclidean norm over grid nodes used by the evaluation code.

| Test | Model | Maximum nodal error | relative_L2 |
| --- | --- | ---: | ---: |
| Mixed-frequency | Plan A | 0.879698 | 0.652677 |
| Mixed-frequency | Plan B | 1.025588 | 0.572910 |
| Polynomial | Plan A | 0.066045 | 0.047870 |
| Polynomial | Plan B | 0.059508 | 0.047007 |

The saved results show better generalization to the smooth polynomial example, with larger errors for the test containing frequencies outside the training range.

## Validation

Verified on Windows with a fresh Python 3.12.14 environment and PyTorch 2.11.0 (CPU):

- Dependency installation and `python -m pip check` passed.
- `python evaluate.py --check` reproduced all four reference evaluation rows.
- The default `main2.ipynb` ran from start to finish.
- Both architectures completed a two-step training smoke test on a reduced grid, including checkpoint saving, reloading, prediction, and plotting. The smoke test used Ngrid=6, batch size 2, and smaller networks.
- Included model files and the reference CSV were unchanged by validation.

The full 10,001-step training run, GPU execution, and macOS/Linux execution have not been revalidated. Short training checks verify the execution path, not final training accuracy.

## Known limitations and items to reconcile

- The maximum nodal errors in the paper abstract differ from `test_results.csv`. The paper and presentation are preserved as supplied; use the CSV and table above for the saved evaluation values pending reconciliation.
- The paper describes a training Fourier cutoff of K=4. The current generator uses `config.RANKS=5` for the first sample in each batch and a hard-coded cutoff of 4 for subsequent samples. This setting is separate from Plan A's factorization rank of 50.
- The factorized implementation still explicitly constructs the four-dimensional kernel tensor; it does not yet implement a fully memory-efficient factorized quadrature contraction.
- Stored notebook outputs have been cleared. Saved model files and result images are included separately.
- No open-source license has been selected. The author should choose whether to add a `LICENSE` file before release.
