import numpy as np
from scipy import linalg
from utils import *
import copy

def hat_map(theta):
    '''
    :param theta: angle of rotation  3*1
    :return: skew_symmetric matrix   3*3
    '''
    theta1 = theta[0]
    theta2 = theta[1]
    theta3 = theta[2]
    theta_hat = np.array([[0, -theta3, theta2], [theta3, 0, -theta1], [-theta2, theta1, 0]])
    return theta_hat

def twist(xi):
    '''
    :param xi: 6*1  [rho, theta]'
    :return: twist matrix  4*4
    '''
    rho = xi[0:3]                                    # 3*1
    theta_hat = hat_map(xi[3:])                      # 3*3
    temp = np.column_stack((theta_hat, rho))            # 3*4
    xi_twist =  np.row_stack((temp, np.zeros((1, 4))))  # 4*4
    return xi_twist

def curly_wedge(u_t):
    '''
    :param u_t: input at time t   [v_t, omega_t]', 6*1
    :return: curly_wedge of u_t   6*6
    '''
    omega_t_hat = hat_map(u_t[3:])                          # 3*3
    v_t_hat = hat_map(u_t[0:3])                              # 3*3
    temp1 = np.column_stack((omega_t_hat, v_t_hat))             # 3*6
    temp2 = np.column_stack((np.zeros((3, 3)), omega_t_hat))    # 3*6
    u_t_curly = np.row_stack((temp1, temp2))                    # 6*6
    return u_t_curly

def prediction(u_t, tau, mu_t_t, sigma_t_t, W):
    '''
    :param u_t: input at time t   [v_t, omega_t]', 6*1
    :param tau: time discretization                1
    :param mu_t_t: mean of T_t at time t           4*4
    :param sigma_t_t: covariance of T_t at time t  6*6
    :param W: covariance of noise w_t at time t    6*6
    :return: mu_tp1_t    4*4         sigma_tp1_t    6*6
    '''
    mu_tp1_t = np.dot(linalg.expm(-tau * twist(u_t)), mu_t_t)    # 4*4
    u_t_curly = curly_wedge(u_t)    # 6*6
    sigma_tp1_t = np.dot(np.dot(linalg.expm(-tau * u_t_curly), sigma_t_t), linalg.expm(-tau * u_t_curly).T) + (tau**2) * W  # 6*6
    return mu_tp1_t, sigma_tp1_t


