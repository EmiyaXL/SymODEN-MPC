import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from symoden import SymODEN_R
from data import \
    get_dataset, arrange_data, get_data
from utils import L2_loss

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("--- new folder ---")
    else:
        print("--- There is the folder ---")


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=6, type=int, help='dimensionality of input tensor')
    parser.add_argument('--learn_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='relu', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=50, type=int, help='number of gradient steps between prints')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--size', type=int, default=1)
    parser.add_argument('--win', type=int, default=100)
    parser.add_argument('--if_nor', type=int, default=1)
    parser.add_argument('--h_size', type=int, default=256)
    parser.add_argument('--g_size', type=int, default=128)
    parser.add_argument('--time_step', type=int, default=0.001)
    parser.add_argument('--init', type=int, default=4000)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--num_points', type=int, default=100,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--batch', type=int, default=120, help='batch size')
    parser.add_argument('--path', default='demo', type=str, help='path')
    parser.add_argument('--file', default='demo', type=str, help='file name')
    parser.add_argument('--solver', default='euler', type=str, help='type of ODE Solver for Neural ODE')

    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


if __name__ == "__main__":
    from torchdiffeq import odeint_adjoint as odeint
    from tqdm import tqdm

    args = get_args()
    trainornot = args.train

    PATH = '/* Source Path */' + args.file

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init model and optimizer
    H_net = MLP(args.input_dim, args.h_size, 1, args.nonlinearity).to(device)
    g_net = MLP(3, args.g_size, 3).to(device)
    model = SymODEN_R(args.input_dim, H_net=H_net, g_net=g_net, device=device, baseline=False)

    size = args.size
    # calculate loss mean and std for each traj.
    data1, state_mu, state_std = get_data(timesteps=args.num_points * args.batch // size, batch=size,
                                          init=args.init,
                                          path=args.path, if_nor=args.if_nor,
                                          win=args.num_points * args.batch // size)
    train_x1, t_eval1 = arrange_data(data1['x'], data1['t'], num_points=args.num_points * args.batch // size)
    train_x1 = torch.tensor(train_x1, requires_grad=True, dtype=torch.float32).to(device)
    t_eval1 = torch.tensor(t_eval1, requires_grad=True, dtype=torch.float32).to(device)
    state_mu = torch.tensor(state_mu, requires_grad=True, dtype=torch.float32).to(device)
    state_std = torch.tensor(state_std, requires_grad=True, dtype=torch.float32).to(device)
    model.load_state_dict(torch.load(PATH))

    train_loss = []
    test_loss = []
    tmp_x_hat = []
    u_tmp = []
    for i in range(train_x1.shape[0]):
        train_x_hat = []
        with torch.no_grad():
            for j in tqdm(range(train_x1.shape[1] - 1)):
                model.get_u(train_x1[i, j, :, -1:])
                if j == 0:
                    tmp_x_hat = odeint(model, train_x1[i, j, :, :-1], torch.tensor([0, args.time_step]),
                                       method=args.solver)
                    train_x_hat.append(tmp_x_hat[0])
                else:
                    t1 = time.time()
                    tmp_x_hat = odeint(model, tmp_x_hat[-1],
                                       torch.tensor([args.time_step * j, args.time_step * (j + 1)]),
                                       method=args.solver)
                    # print("time:", time.time()-t1)
                train_x_hat.append(tmp_x_hat[-1])
        train_x_hat = torch.stack(train_x_hat, dim=0)

    y = torch.empty(args.num_points * args.batch, 6)
    y_hat = torch.empty(args.num_points * args.batch, 6)
    with torch.no_grad():
        for i in range(train_x1.shape[2]):
            for j in range(train_x1.shape[1]):
                if args.if_nor:
                    y[i * train_x1.shape[1] + j, :] = (train_x1[0, j, i, :-1] * state_std[:]) + state_mu[:]
                    y_hat[i * train_x1.shape[1] + j, :] = (train_x_hat[j, i, :] * state_std[:]) + state_mu[:]
                else:
                    y[i * args.num_points * args.batch + j, :] = train_x1[0, j, i, :-1]
                    y_hat[i * args.num_points * args.batch + j, :] = train_x_hat[j, i, :-1]
    ts = y.shape[0]
    x = np.linspace(1, ts, ts) * 0.001
    yn = np.stack([y, y_hat], axis=1)
    for l in range(6):
        plt.subplot(2, 3, l + 1)
        plt.plot(yn[:, :, l])
    plt.show()
