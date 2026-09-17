import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# u = lambda x, y: torch.sin(x*torch.pi) * torch.sin(y*torch.pi)
# f = lambda x, y: -2*torch.pi**2 * torch.sin(x*torch.pi) * torch.sin(y*torch.pi)

a, b, c = torch.distributions.Uniform(1,2).sample((1,1)).item(), torch.distributions.Uniform(1,2).sample((1,1)).item(), torch.distributions.Uniform(1,2).sample((1,1)).item()
v = lambda x,y: a*(((x+1)/2)**4 - 6*((x+1)/2)**2*((y+1)/2)**2 + ((y+1)/2)**4) + b*((x+1)/2)**4 + c*((y+1)/2)**4

v1 = lambda y: 3*a*(y+1)**2/4 + 3*c*(y+1)**2
v2 = lambda y: -3*a*(4-(y+1)**2)/4 + 3*c*(y+1)**2

v3 = lambda x: 3*a*(x+1)**2/4 + 3*b*(x+1)**2
v4 = lambda x: 3*a*((x+1)**2-4)/4 + 3*b*(x+1)**2

f = lambda x,y: 12*(c*(y+1)**2 + b*(x+1)**2)/4 + \
    (-(1-x)/2*v1(y) - (1+x)/2*v2(y) - (1-y)/2*v3(x) - (1+y)/2*v4(x))


w = lambda x, y: (
    - (1 - x) / 2 * v(torch.tensor(-1), y)                        # 上邊界 x = -1
    - (1 + x) / 2 * v(torch.tensor( 1), y)                        # 下邊界 x = 1
    - (1 - y) / 2 * v(x, torch.tensor(-1))                        # 左邊界 y = -1
    - (1 + y) / 2 * v(x, torch.tensor( 1))                        # 右邊界 y = 1
    + (1 - x) * (1 - y) / 4 * v(torch.tensor(-1), torch.tensor(-1))             # 左上角
    + (1 - x) * (1 + y) / 4 * v(torch.tensor(-1), torch.tensor( 1))             # 右上角
    + (1 + x) * (1 - y) / 4 * v(torch.tensor( 1), torch.tensor(-1))             # 左下角
    + (1 + x) * (1 + y) / 4 * v(torch.tensor( 1), torch.tensor( 1))             # 右下角
)

u = lambda x, y: v(x,y) + w(x,y)
# model_case = []
# Ngrid_list = [22]
# mu_list  = [0]
# ffsize_list = [500]
# ffscale_list = [0.5]
Batch_Size = 16
Validate_Batch_size = 1

# for Ngrid in Ngrid_list:
#     for mu in mu_list:
#         for ffsize in ffsize_list:
#             for ffscale in ffscale_list:
#                 model_case.append(
#                     (Ngrid, [4, 50, 50, 1], nn.GELU(), ffsize, ffscale, mu)
#                 )

model_case = [(22, [4, 50, 50, 1], nn.GELU(), 100, 1, 0)]
# model_case = [(22, [2, 50, 50, 50], nn.GELU(), 100, 1, 0)]

TRAIN  = False
USE_FF = True
USE_NORM = False
Lambda = 1e-4 # r1^2 +\lambda r2^2
Change_data = 100
TOLERANCE = 1e-16
RANKS = 5 # rank of Fourier basis

# iteration counts and check
Adam_iter = 10001
tr_iter_max    = 0 #1001                     # max. iteration
ts_input_new   = 100                       # renew testing points 
ts_over_tr_tol = 100.

# early stop
early_stop_patience = 4000   # 幾步沒有進步就停
min_delta = 1e-2           # 要進步超過這個量，才算真的進步




