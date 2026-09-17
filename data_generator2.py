from grid import gLLNodesAndWeights
from diff import DiffMat
from config import *
import torch
torch.set_default_dtype(torch.float64)

def generate_smooth_U_fourier(
    X2d, Y2d,
    S2d, T2d,
    num_modes=8,
    alpha=2.0,
    normalize=True,
):
    """
    Gaussian Random Fourier Field

    u(x,y)=Σ a_mn sin(mπ(x+1)/2) sin(nπ(y+1)/2)

    Parameters
    ----------
    num_modes : int
        最大 Fourier mode
    alpha : float
        頻譜衰減速度
            1   -> 粗糙
            2   -> 很平滑 (推薦)
            3   -> 非常平滑
    """

    device = X2d.device
    dtype = X2d.dtype

    U_xy = torch.zeros_like(X2d)
    F_xy = torch.zeros_like(X2d)

    U_st = torch.zeros_like(S2d)
    F_st = torch.zeros_like(S2d)

    for kx in range(1, num_modes + 1):
        for ky in range(1, num_modes + 1):

            # Gaussian coefficient
            coeff = torch.randn((), device=device, dtype=dtype)

            # spectral decay
            coeff /= (kx**2 + ky**2)**(alpha/2)

            # basis
            basis_xy = torch.sin(kx * torch.pi * (X2d + 1) / 2) * torch.sin(ky * torch.pi * (Y2d + 1) / 2)

            basis_st = torch.sin(kx * torch.pi * (S2d + 1) / 2) * torch.sin(ky * torch.pi * (T2d + 1) / 2)

            eig = -((kx * torch.pi / 2) ** 2 + (ky * torch.pi / 2) ** 2)

            U_xy += coeff * basis_xy
            F_xy += coeff * eig * basis_xy

            U_st += coeff * basis_st
            F_st += coeff * eig * basis_st

    if normalize:

        scale = U_xy.abs().max()

        if scale > 1e-10:
            U_xy /= scale
            F_xy /= scale

        scale = U_st.abs().max()

        if scale > 1e-10:
            U_st /= scale
            F_st /= scale

    return U_xy, U_st, F_xy, F_st

def u_train_data_generator(Ngrid, N=1):
    if N <= 0: assert False, "N must be a positive integer."
    Nx = Ny = Ns = Nt = Ngrid

    x, wx = gLLNodesAndWeights(Nx); s, ws = gLLNodesAndWeights(Ns)
    y, wy = gLLNodesAndWeights(Ny); t, wt = gLLNodesAndWeights(Nt)

    X2d, Y2d = torch.meshgrid(x, y, indexing='ij')
    W2d = wx[:, None] * wy[None, :]

    x, wx = gLLNodesAndWeights(Nx); s, ws = gLLNodesAndWeights(Ns)
    y, wy = gLLNodesAndWeights(Ny); t, wt = gLLNodesAndWeights(Nt)

    X2d, Y2d = torch.meshgrid(x, y, indexing='ij')
    S2d, T2d = torch.meshgrid(s, t, indexing='ij')

    X4d, Y4d, S4d, T4d = torch.meshgrid((x, y, s, t), indexing='ij')

    D = DiffMat(x); D2 = D@D
    B = torch.zeros_like(D)
    B[0, 0] = 1/(wx[0]**2)
    B[-1, -1] = 1/(wx[-1]**2)
    # D2 = D2 - B

    # --- 生成第 1 個平滑的 U ---
    with torch.no_grad():
        U_xy, U_st, F_xy, F_st = generate_smooth_U_fourier(X2d, Y2d, S2d, T2d, num_modes=RANKS) 
        F = F_xy # D2@U_st + U_st@D2.T

    train_data_xy = torch.stack((X2d, Y2d), axis=2)
    train_data_st = torch.stack((S2d, T2d), axis=2)
    train_data = torch.stack((X4d, Y4d, S4d, T4d), axis=4)
    train_label_f = F
    train_label_u = U_xy
    grid_xy_tmp = torch.stack((X2d, Y2d), axis=2)
    grid_st_tmp = torch.stack((S2d, T2d), axis=2)
    grid_tmp    =  torch.stack((X4d, Y4d, S4d, T4d), axis=4)

    for i in range(N-1):
        train_data_xy = torch.cat((train_data_xy, grid_xy_tmp))
        train_data_st = torch.cat((train_data_st, grid_st_tmp))
        train_data    = torch.cat((train_data, grid_tmp))
        # 循環中生成後續的平滑 U ---
        with torch.no_grad():
            U_xy, U_st, F_xy, F_st = generate_smooth_U_fourier(X2d, Y2d, S2d, T2d, num_modes=4)
            F = F_xy # D2@U_st + U_st@D2.T

        train_label_f = torch.cat((train_label_f, F))
        train_label_u = torch.cat((train_label_u, U_xy))

    # 後續的 reshape 保持不變...
    train_data_xy = train_data_xy.reshape(N, Ngrid+1, Ngrid+1, 2)
    train_data_st = train_data_st.reshape(N, Ngrid+1, Ngrid+1, 2)
    train_data    = train_data.reshape(N, Ngrid+1, Ngrid+1, Ngrid+1, Ngrid+1, 4)
    train_label_f = train_label_f.reshape(N, Ngrid+1, Ngrid+1)
    train_label_u = train_label_u.reshape(N, Ngrid+1, Ngrid+1)

    return ((X2d, Y2d), (wx, wy), W2d, D, D2, U_xy, F), train_data, train_data_xy, train_data_st, train_label_f, train_label_u