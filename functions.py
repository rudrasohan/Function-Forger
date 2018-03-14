import numpy as np
import torch
def distribution(mu, sigma):
	return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n))).cuda()
def random_input():
	return lambda m, n: torch.rand(m, n).cuda()
#print(distribution(0,0.1))
#print(random_input())
#n = 10
#m = 5
#print(torch.Tensor(np.random.normal(0, 0.1, (1, n))).cuda())
#print(torch.rand(m, n).cuda())