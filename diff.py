import torch #type: ignore
import grid

def DiffMat(X):
    N = len(X) - 1
    D = torch.empty((N+1, N+1))

    X_I = X.repeat(N+1).reshape((N+1, N+1))
    X_J = X_I.T

    lgP_I = grid.lgP(N, X).repeat(N+1).reshape((N+1, N+1))
    lgP_J = lgP_I.T

    D = lgP_J / (lgP_I * (X_J- X_I))

    D.fill_diagonal_(0)
    D[0, 0] = -N * (N + 1) / 4
    D[N, N] = N * (N + 1) / 4

    return D