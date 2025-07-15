import autograd
import autograd.numpy as np
import numpy as np

__all__ = [np]

import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle
import gym
import torch
import myenv

def data_normalize(data):
    mu = np.mean(data)
    std = np.std(data, ddof=1)
    return (data - mu) / std, mu, std


def read_data(batch=60, timestep=100, path='demo', init_time=2000, if_nor=1):
    px = np.loadtxt('../data/' + path + '/X coordinate, vehicle origin.txt')
    py = np.loadtxt('../data/' + path + '/Y coordinate, vehicle origin.txt')
    phi = np.loadtxt('../data/' + path + '/Yaw, vehicle.txt')
    vx = np.loadtxt('../data/' + path + '/Longitudinal speed, vehicle.txt')
    vy = np.loadtxt('../data/' + path + '/Lateral speed, vehicle.txt')
    w = np.loadtxt('../data/' + path + '/yaw_rate.txt')
    tor = np.loadtxt('../data/' + path + '/input torque.txt')
    Ix = vx
    Iy = vy
    Iw = w

    # data scale
    Ix = np.array(Ix)
    Ix_scale, Ix_mu, Ix_std = data_normalize(Ix)

    Iy = np.array(Iy)
    Iy_scale, Iy_mu, Iy_std = data_normalize(Iy)

    Iw = np.array(Iw)
    Iw_scale, Iw_mu, Iw_std = data_normalize(Iw)

    px = np.array(px)
    px_scale, px_mu, px_std = data_normalize(px)

    py = np.array(py)
    py_scale, py_mu, py_std = data_normalize(py)

    phi = np.array(phi)
    phi_scale, phi_mu, phi_std = data_normalize(phi)

    u = np.array(tor)
    u_scale, u_mu, u_std = data_normalize(u)

    if if_nor:
        p = [Ix_scale[init_time:init_time + batch * timestep], Iy_scale[init_time:init_time + batch * timestep],
             Iw_scale[init_time:init_time + batch * timestep]]
        q = [px_scale[init_time:init_time + batch * timestep], py_scale[init_time:init_time + batch * timestep],
             phi_scale[init_time:init_time + batch * timestep]]
        u = u_scale[init_time:init_time + batch * timestep]
    else:
        p = [Ix[init_time:init_time + batch * timestep], Iy[init_time:init_time + batch * timestep],
             Iw[init_time:init_time + batch * timestep]]
        q = [px[init_time:init_time + batch * timestep], py[init_time:init_time + batch * timestep],
             phi[init_time:init_time + batch * timestep]]
        u = u[init_time:init_time + batch * timestep]

    mu = [px_mu, py_mu, phi_mu, Ix_mu, Iy_mu, Iw_mu]
    std = [px_std, py_std, phi_std, Ix_std, Iy_std, Iw_std]
    x_temp = []
    xs = []

    for k in range(batch):
        for m in range(timestep):
            x_temp.append(
                [q[0][k * timestep + m], q[1][k * timestep + m], q[2][k * timestep + m],
                 p[0][k * timestep + m], p[1][k * timestep + m], p[2][k * timestep + m], u[k * timestep + m]])
        xs.append(x_temp)
        x_temp = []
    xs_f = np.stack(xs, axis=1)

    return xs_f, mu, std


def get_dataset(samples=50, test_split=0.5, timesteps=20):
    data = {}
    xs = read_data()
    data['x'] = np.array([xs])  # (1, 120, 50, 7)
    split_ix = int(samples * test_split)
    split_data = {}
    split_data['x'], split_data['test_x'] = data['x'][:, :, :split_ix, :], data['x'][:, :, split_ix:, :]
    data = split_data
    data['t'] = np.linspace(1, timesteps, timesteps) * 0.001
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points >= 2 and num_points <= len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points - 1:
            x_stack.append(x[:, i:-num_points + i + 1, :, :])
        else:
            x_stack.append(x[:, i:, :, :])
    x_stack = np.stack(x_stack, axis=1)

    x_stack = np.reshape(x_stack,
                         (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval
