import numpy as np
import torch 
from torch.autograd import Variable

(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

def extract(v):
	return v.data.storage().tolist()

def stats(d):
	return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
	mean = torch.mean(data.data, 1, keepdim=True)
	mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
	diffs = torch.pow(data - Variable(mean_broadcast).cuda(), exponent)
	return torch.cat([data, diffs], 1)