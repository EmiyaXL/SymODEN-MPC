import numpy as np
import pandas as pd
import pickle as pkl
import torch
from numpy import random as nr

f = open('/*Path + File name */', 'rb')

data_torch = torch.load(f, map_location='cpu')

i = 0
for c in data_torch.keys():
    print(c)
H_para = data_torch['H_net.linear1.weight'].numpy()
np.savetxt('H_net_linear1_weight.txt', H_para, fmt='%1.4e')
H_para = data_torch['H_net.linear1.bias'].numpy()
np.savetxt('H_net_linear1_bias.txt', H_para, fmt='%1.4e')
H_para = data_torch['H_net.linear2.weight'].numpy()
np.savetxt('H_net_linear2_weight.txt', H_para, fmt='%1.4e')
H_para = data_torch['H_net.linear2.bias'].numpy()
np.savetxt('H_net_linear2_bias.txt', H_para, fmt='%1.4e')
H_para = data_torch['H_net.linear3.weight'].numpy()
np.savetxt('H_net_linear3_weight.txt', H_para, fmt='%1.4e')
H_para = data_torch['H_net.linear3.bias'].numpy()
np.savetxt('H_net_linear3_bias.txt', H_para, fmt='%1.4e')
G_para = data_torch['g_net.linear1.weight'].numpy()
np.savetxt('g_net_linear1_weight.txt', G_para, fmt='%1.4e')
G_para = data_torch['g_net.linear1.bias'].numpy()
np.savetxt('g_net_linear1_bias.txt', G_para, fmt='%1.4e')
G_para = data_torch['g_net.linear2.weight'].numpy()
np.savetxt('g_net_linear2_weight.txt', G_para, fmt='%1.4e')
G_para = data_torch['g_net.linear2.bias'].numpy()
np.savetxt('g_net_linear2_bias.txt', G_para, fmt='%1.4e')
G_para = data_torch['g_net.linear3.weight'].numpy()
np.savetxt('g_net_linear3_weight.txt', G_para, fmt='%1.4e')
G_para = data_torch['g_net.linear3.bias'].numpy()
np.savetxt('g_net_linear3_bias.txt', G_para, fmt='%1.4e')