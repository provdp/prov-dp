import torch as th

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
