import torch #type: ignore

def lgP (n, xi):
  """
  Evaluates P_{n}(xi) using an iterative algorithm
  """
  if n == 0:
    
    return torch.ones (xi.size())
  
  elif n == 1:
    
    return xi

  else:

    fP = torch.ones (xi.size()); sP = xi.clone(); nP = torch.empty (xi.size())

    for i in range (2, n + 1):

      nP = ((2 * i - 1) * xi * sP - (i - 1) * fP) / i

      fP = sP; sP = nP

    return nP

def dLgP (n, xi):
  """
  Evaluates the first derivative of P_{n}(xi)
  """
  return n * (lgP (n - 1, xi) - xi * lgP (n, xi))\
           / (1 - xi ** 2)

def d2LgP (n, xi):
  """
  Evaluates the second derivative of P_{n}(xi)
  """
  return (2 * xi * dLgP (n, xi) - n * (n + 1)\
                                    * lgP (n, xi)) / (1 - xi ** 2)

def d3LgP (n, xi):
  """
  Evaluates the third derivative of P_{n}(xi)
  """
  return (4 * xi * d2LgP (n, xi)\
                 - (n * (n + 1) - 2) * dLgP (n, xi)) / (1 - xi ** 2)

def gLLNodesAndWeights (n, epsilon = 1e-15):
  """
  Computes the GLL nodes and weights
  """
  n = n + 1
  if n < 2:
    
    print ('Error: n must be larger than 1')
  
  else:
    
    x = torch.empty (n)
    w = torch.empty (n)
    
    x[0] = -1; x[n - 1] = 1
    w[0] = w[0] = 2.0 / ((n * (n - 1))); w[n - 1] = w[0]
    
    n_2 = n // 2
    
    for i in range (1, n_2):
      
      xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) *\
           torch.cos (torch.tensor(4 * i + 1) * torch.pi / (4 * (n - 1) + 1))
      
      error = 1.0
      
      while error > epsilon:
        
        y  =  dLgP (n - 1, xi)
        y1 = d2LgP (n - 1, xi)
        y2 = d3LgP (n - 1, xi)
        
        dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
        
        xi -= dx
        error = abs (dx)
      
      x[i] = -xi
      x[n - i - 1] =  xi
      
      w[i] = 2 / (n * (n - 1) * lgP (n - 1, x[i]) ** 2)
      w[n - i - 1] = w[i]

    if n % 2 != 0:

      x[n_2] = 0
      w[n_2] = 2.0 / ((n * (n - 1)) * lgP (n - 1, x[n_2]) ** 2)
      
  return x, w