# Toy example
import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)


wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
xbow2


smokk = torch.tril(torch.ones((T, T)))
wei = torch.zeros((T, T))
wei = wei.masked_fill(smokk == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
