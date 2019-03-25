import numpy as np
from scipy import linalg
from utils import *
import copy

def projection(q):
    '''
    :param q: parameters 4*1
    :return: projection  4*1
    '''
    q3 = q[2]
    proj = q/q3
    return proj

def proj_derivative(q):
    '''
    :param q: parameters 4*1
    :return: projection derivative  4*1
    '''
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    proj_deri = 1/q3 * np.array([[1, 0, -q1/q3, 0], [0, 1, -q2/q3, 0], [0, 0, 0, 0], [0, 0, -q4/q3, 1]])
    return proj_deri

def cal_mu_ini(M, T_i2c, T_t, z_t_i):
    '''
    :param M: calibration matrix      4*4
    :param T_i2c: imu to camera
    :param T_t: T_w2i   world to imu
    :param z_t_i: At time t, the first observation i whose features are not [-1,-1,-1,-1]'  4*1
    :return: mu_t_j                 initial map at time t
    '''
    u_L = z_t_i[0]
    v_L = z_t_i[1]
    u_R = z_t_i[2]
    v_R = z_t_i[3]
    c_u = M[0, 2]
    c_v = M[1, 2]
    fs_u = M[0, 0]
    fs_v = M[1, 1]
    fs_ub = -M[2, 3]
    z = fs_ub / (u_L - u_R)
    mu_t_j = z * np.dot(np.linalg.pinv(M), z_t_i)
    T_c2w = np.linalg.inv( np.dot(T_i2c, T_t) )  # T_i2c*T_t---world to camera  4*4
    mu_t_j = np.dot(T_c2w, mu_t_j)   # 4*1
    return mu_t_j

def landmark_update(H_i_j_t, mu_t_j, sigma_t_j, V, D, z_t_i):
    '''
    :param H_i_j_t: 4*3      for one landmark
    :param mu_t_j: 4*1     for one landmark, j = 1,2,.....m
    :param sigma_t_j: 3*3    for one landmark
    :param V: covariance of observation noise  4*4
    :param D: dilation matrix   4*3
    :param z_t_i: observation whose feature is not [-1, -1, -1, -1]', total 4*N_t, for one landmark
    :return: mu_tp1_j    4*1      sigma_tp1_j    3*3
    '''
    temp1 = np.dot( np.dot(T_i2c, T_t), mu_t_j)
    z_t_i_hat = np.dot(M, projection(temp1))
    temp2 = np.dot( np.dot(H_i_j_t, sigma_t_j), H_i_j_t.T ) + V
    K_i_j_t = np.dot( np.dot(sigma_t_j, H_i_j_t.T), np.linalg.inv(temp2) ) # 3*4
    mu_tp1_j = mu_t_j + np.dot( np.dot(D, K_i_j_t), (z_t_i - z_t_i_hat) )
    sigma_tp1_j = np.dot( ( np.identity(3) - np.dot(K_i_j_t, H_i_j_t) ), sigma_t_j)
    return mu_tp1_j, sigma_tp1_j
