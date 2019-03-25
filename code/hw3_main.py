import numpy as np
from utils import *
from prediction import *
from mapping import *

if __name__ == '__main__':

    filename = "./data/0042.npz"
    # t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

    t, features, v_t, omega_t, K, b, T_i2c = load_data(filename)
    M = np.array([[K[0, 0], 0, K[0, 2], 0], [0, K[1, 1], K[1, 2], 0], [K[0, 0], 0, K[0,2], -b*K[0, 0]], [0, K[1, 1], K[1, 2], 0]])   # calibration matrix M made up of intrinsic parameters and baseline, 4*4


    # (a) IMU Localization via EKF Prediction
    u = np.row_stack((v_t, omega_t))  # input u as time t, 6*1106 for 0027.npz, 6*500 for 0042.npz
    tau = t[0, 1:] - t[0, 0:-1]  # 1*1105 or 1*499
    mu_t_t = np.identity(4)  # initial pose mu_t_t, identity matrix   4*4
    sigma_t_t = np.identity(6)  # initial pose covariance sigma_t_t, identity matrix   6*6
    W = np.identity(6)  # 6*6, set to be identity matrix
    mu = np.zeros((4, 4, u.shape[1]))     # 4*4*1106 for 0027.npz, 4*4*500 for 0042.npz
    sigma = np.zeros((6, 6, u.shape[1]))  # 6*6*1106 for 0027.npz, 6*6*500 for 0042.npz
    T_i2w = np.zeros((4, 4, u.shape[1]))      # 4*4*1106 for 0027.npz, 4*4*500 for 0042.npz
    T_i2w[:, :, 0] = np.linalg.inv(copy.deepcopy(mu_t_t)) # 4*4*1106 for 0027.npz, 4*4*500 for 0042.npz, pose from imu to world
    mu[:, :, 0] = copy.deepcopy(mu_t_t)
    sigma[:, :, 0] = copy.deepcopy(sigma_t_t)
    for t in range(1, u.shape[1]):
        u_t = u[:, t]  # 6*1
        tau_t = tau[t - 1]  # 1
        mu_tp1_t, sigma_tp1_t = prediction(u_t, tau_t, mu_t_t, sigma_t_t, W)
        mu[:, :, t] = copy.deepcopy(mu_tp1_t)
        sigma[:, :, t] = copy.deepcopy(sigma_tp1_t)
        T_i2w[:, :, t] = np.linalg.inv(copy.deepcopy(mu_tp1_t))  # pose
        mu_t_t = copy.deepcopy(mu_tp1_t)
        sigma_t_t = copy.deepcopy(sigma_tp1_t)

	# (b) Landmark Mapping via EKF Update
    time = features.shape[2]       # time
    m = features.shape[1]          # the # of landmark
    temp1 = np.array([-1, -1, -1, -1])  # to determine whether the feature is observed
    sigma_t_j = np.identity(3)     # 3*3
    V = np.identity(4)             # 4*4
    D = np.row_stack( ( np.identity(3), np.zeros((1, 3)) ) )   # dilation matrix   4*3
    sigma_t = np.zeros((3*m, 3*m))   # 3m*3m
    mu_t = np.zeros((4, m))          # 4*m
    for j in range(m):
        for t in range(time):
            feature = copy.deepcopy(features[:, j, t])
            count = 0  # if equal to 0, need to be initialized
            if sum( (feature != temp1) ) == 4:
                if count == 0:
                    z_t_i = copy.deepcopy(feature)
                    mu_t_j = copy.deepcopy(cal_mu_ini(M, T_i2c, mu[:, :,t],  z_t_i))
                    count += 1  # if equal to 1, not need to be initialized
                    # import pdb
                    # pdb.set_trace()
                    # print(mu_t_j)
                else:
                    z_t_i = copy.deepcopy(feature)
                    temp2 = np.dot( np.dot(T_i2c, mu[:, :, t]), mu_t_j )
                    print(temp2)
                    temp3 = np.dot( np.dot(proj_derivative(temp2), T_i2c), mu[:, :, t] )
                    H_i_j_t = np.dot( np.dot(M, temp3), D)
                    mu_tp1_j, sigma_tp1_j = landmark_update(H_i_j_t, mu_t_j, sigma_t_j, V, D, z_t_i)
                    mu_t_j = copy.deepcopy(mu_tp1_j)
                    sigma_t_j = copy.deepcopy(sigma_tp1_j)
        mu_t[:, j] = mu_t_j.T
        sigma_t[3*j:3*(j + 1), 3*j:3*(j+1)] = sigma_t_j

    map = np.zeros((2, m))
    map = mu_t[0:2, :]
    visualize_trajectory_2d(T_i2w, map, path_name=filename, show_ori=False)

	# (c) Visual-Inertial SLAM (Extra Credit)

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
