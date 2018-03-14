import torch 
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import Gen_Net, Disc_Net
from functions import *
from utils import *

mu = 0
sigma = 0.1
g_i = 1
g_h = 50
g_o = 1
d_i = 100
d_h = 50
d_o = 1
minibatch_size = d_i
g_lr = 2e-4
d_lr = 2e-4
num_epochs = 30000
print_interval = 200
d_steps = 1
g_steps = 1

fn_real = distribution(mu, sigma)
fn_fake = random_input()
G = Gen_Net(input_size=g_i,hidden_size=g_h,output_size=g_o).cuda()
D = Disc_Net(input_size=d_input_func(d_i),hidden_size=d_h,output_size=d_o).cuda()
criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas = (0.9,0.999))
d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas = (0.9,0.999))

for epoch in range(num_epochs):
	for index in range(d_steps):
		D.zero_grad()
		d_real_data = Variable(fn_real(d_i)).cuda()
		d_real_descision = D(preprocess(d_real_data))
		d_real_error = criterion(d_real_descision, Variable(torch.ones(1)).cuda())
		d_real_error.backward()
		d_gen_input = Variable(fn_fake(minibatch_size, g_i)).cuda()
		d_fake_data = G(d_gen_input).detach()
		d_fake_descision = D(preprocess(d_fake_data.t()))
		d_fake_error = criterion(d_fake_descision, Variable(torch.zeros(1)).cuda())
		d_fake_error.backward()
		d_optimizer.step()
	
	for index in range(g_steps):
		G.zero_grad()

		gen_input = Variable(fn_fake(minibatch_size, g_i)).cuda()
		g_fake_data = G(gen_input)
		dg_fake_descision = D(preprocess(g_fake_data.t()))
		g_error = criterion(dg_fake_descision, Variable(torch.ones(1)).cuda())
		g_error.backward()
		g_optimizer.step()

	if epoch % print_interval == 0:
		print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,extract(d_real_error)[0],extract(d_fake_error)[0],
			extract(g_error)[0],stats(extract(d_real_data)),stats(extract(d_fake_data))))

