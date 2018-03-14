import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gen_Net(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Gen_Net, self).__init__()
		self.fc1 = nn.Linear(input_size,hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		x = self.fc3(x)
		return x

class Disc_Net(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Disc_Net, self).__init__()
		self.fc1 = nn.Linear(input_size,hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.selu(self.fc2(x))
		x = F.sigmoid(self.fc3(x))
		return x

