import numpy as np
import matplotlib.pyplot as plt
import math
import utils

from EKF import ExtendedKalmanFilter

def F_fun(x, dt=None):
    px, py, v, si, si_dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]
    
    e1 = (v/si_dot) * ( math.sin(si+si_dot*dt) - math.sin(si))
    e2 = (v/si_dot) * (-math.cos(si+si_dot*dt) + math.cos(si))
    e4 = si_dot*dt

    F = np.array([e1, e2, 0, e4, 0]).reshape(5, 1)
    new_x = x + F
    return new_x

def F_Jacobian_fun(x, z_dim, x_dim, dt):
    px, py, v, si, si_dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]

    sin1 = math.sin(si)
    cos1 = math.cos(si)
    sin2 = math.sin(si+si_dot*dt)
    cos2 = math.cos(si+si_dot*dt)
    
    e13 = (1/si_dot) * ( sin2 - sin1)
    e14 = (v/si_dot) * ( cos2 - cos1)
    e15 = - (v/si_dot**2) * (sin2 - sin1) + (v/si_dot)*dt*cos1

    e23 = (1/si_dot) * (-cos2 + cos1)
    e24 = (v/si_dot) * ( sin2 - sin1)
    e25 = - (v/si_dot**2) * (-cos2 + cos1) + (v/si_dot)*dt*sin1

    e45 = dt

    H_ = np.array([[  1,   0, e13, e14, e15],
                   [  0,   1, e23, e24, e25],
                   [  0,   0,   1,   0,   0],
                   [  0,   0,   0,   1, e45],
                   [  0,   0,   0,   0,   1]])

    return H_

def Q_update(x, process_noise_cov, dt):
    px, py, v, si, si_dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]

    fq = np.array([[(dt**2/2)*math.cos(si),       0],
                   [(dt**2/2)*math.sin(si),       0],
                   [                    dt,       0],
                   [                     0, dt**2/2],
                   [                     0,      dt]])

    Q = np.dot(np.dot(fq, process_noise_cov), fq.T)
    return Q

def H_fun(x, dt=None):
    px, py, v, si, si_dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]
    vx = v*math.cos(si)
    vy = v*math.sin(si)

    rho = math.sqrt(px**2 + py**2)
    if(math.fabs(rho) < 0.0001):
        print("ProcessMeasurement () - Error - Division by Zero")
        phi = math.atan2(py/px)
        ro_dot = 0
    else:
        phi = math.atan2(py, px)
        ro_dot = (px*vx+py*vy) / rho
    
    return np.array([rho, phi, ro_dot]).reshape(3,1)

def H_Jacobian_fun(x, z_dim, x_dim, dt=None):
    px, py, v, si, si_dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0]
    vx = v*math.cos(si)
    vy = v*math.sin(si)

    H = np.zeros((z_dim, x_dim))
    ro = np.sqrt(px**2 + py**2)
    ro_2 = ro**2
    
    H[0, 0] = px / ro
    H[0, 1] = py / ro

    H[1, 0] = -py / ro_2
    H[1, 1] =  px / ro_2


    # temp = (px*math.cos(si) + py*math.sin(si)) / ro_2

    # H[2, 0] = (v/ro) * (math.cos(si) - px*temp)
    # H[2, 1] = (v/ro) * (math.sin(si) - py*temp)
    H[2, 0] = (v/ro) - (math.cos(si) - (px*px*math.cos(si)/ro_2) - (px*py*math.sin(si)/ro_2))
    H[2, 1] = (v/ro) - (math.sin(si) - (py*py*math.sin(si)/ro_2) - (px*py*math.cos(si)/ro_2))
    H[2, 2] = (px*math.cos(si) + py*math.sin(si)) / ro
    H[2, 3] = (v/ro) * (py*math.cos(si) - px*math.sin(si))

    return H

if __name__ == '__main__':
    z_dim = 3
    x_dim = 5
    
    a_noise_std = 9
    si_ddot_noise_std = 9

    ro_noise_std = 0.9
    si_noise_std = 0.009
    ro_dot_noise_std = 0.9

    process_noise_covariance = np.array([[a_noise_std**2,                     0],
                                         [0              , si_ddot_noise_std**2]])

    measument_noise_covariance = np.array([[ro_noise_std**2,               0,                   0],
                                           [              0, si_noise_std**2,                   0],
                                           [              0,               0, ro_dot_noise_std**2]]) 

    EKF_tracker = ExtendedKalmanFilter(x_dim, z_dim, F_fun=F_fun, H_fun=H_fun, F_Jacobian_fun=F_Jacobian_fun, 
                                       H_Jacobian_fun=H_Jacobian_fun, Q_update_fun=Q_update, 
                                       process_noise_cov=process_noise_covariance, measument_noise_cov=measument_noise_covariance)

    ################################## Read the sensor Measurments ##################################
    M = utils.Measurments('./data/sample-laser-radar-measurement-data-1.txt')
    radar_measurments = M.radar_measurments
    radar_groundtruth = M.radar_ground_truth
    
    ################################## Run EKF Tracker ###############################################
    estimations = []
    Innovation = []
    i = 1
    EKF_tracker.prev_time_stamp = radar_measurments[0][z_dim]
    for z in radar_measurments:
        t = z[z_dim]
        EKF_tracker.dt = (t - EKF_tracker.prev_time_stamp) / 1000000.0
        EKF_tracker.prev_time_stamp = t

        z = np.array(z[:z_dim]).reshape(z_dim, 1)

        EKF_tracker.predict()
        v = EKF_tracker.update(z)

        Innovation.append(v)
        estimations.append(EKF_tracker.x)

    ################################## Evaluate the performance ##################################
    radar_groundtruth = np.array(radar_groundtruth)
    estimations = np.array(estimations)
    Innovation = np.array(Innovation)
    radar_groundtruth = np.expand_dims(radar_groundtruth, axis=2)

    # RMSE = EKF_tracker.calculate_rmse(estimations, radar_groundtruth).T
    # print(RMSE)

    fig = plt.figure(figsize = [16,6])
    X = np.arange(1, len(Innovation)+1, 1)
    
    plt.subplot(1, 3, 1)
    ax = plt.scatter(x=X.reshape(-1), y=Innovation[:, 0])
    plt.title('Innovation of radial distance')
    plt.xlabel('No. measurments')
    plt.ylabel('Innovation')
    
    plt.subplot(1, 3, 2)
    ax = plt.scatter(x=X.reshape(-1), y=Innovation[:, 1])
    plt.title('Innovation of si')
    plt.xlabel('No. measurments')
    plt.ylabel('Innovation')
    
    plt.subplot(1, 3, 3)
    ax = plt.scatter(x=X.reshape(-1), y=Innovation[:, 2])
    plt.title('Innovation of  radial distance rate')
    plt.xlabel('No. measurments')
    plt.ylabel('Innovation')
    plt.savefig('./assert/EKF_CTRV_radar_Innovation.png')
    # plt.show()

    fig = plt.figure(figsize = [8,8])
    plt.scatter(radar_groundtruth[:, 0], radar_groundtruth[:, 1], label='Ground Truth Measurment')
    plt.scatter(estimations[:, 0], estimations[:, 1], label='Kalman Filter Estimation')
    plt.title('Ground Truth Vs EKF Estimation on LiDAR Data')
    plt.xlabel('X (position)')
    plt.ylabel('Y (Position)')
    plt.legend()
    plt.savefig('./assert/EKF_CTRV_radar.png')
    plt.show()