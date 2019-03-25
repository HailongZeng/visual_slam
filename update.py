import numpy as np
from scipy import linalg
from utils import *
from prediction import *
import copy

def update(mu_tp1_t, sigma_tp1_t, M, T_i2c, map, z_tp1):
    '''
    :param mu_tp1_t: mu_tp1_t from prediction        4*4
    :param sigma_tp1_t: sigma_tp1_t from prediction  4*4
    :param M: calibration matrix                     4*4
    :param T_i2c: imu to camera                      4*4
    :param map: map at time t, from part(b)          4*m
    :param z_tp1: new observation at t+1             4*N_t
    :return: mu_tp1_tp1   4*4         sigma_tp1_tp1    6*6
    '''
    z_tp1_