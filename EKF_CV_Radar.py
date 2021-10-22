import numpy as np
import matplotlib.pyplot as plt
import math
import utils

from EKF import ExtendedKalmanFilter

def F_fun(x, dt=None):
    F = np.array([[1, 0, dt,  0],
                  [0, 1,  0, dt],
                  [0, 0,  1,  0],
                  [0, 0,  0,  1]])
    return F

def Q_update(process_noise_cov, dt):
    fq = np.array([[dt**2/2,       0],
                   [      0, dt**2/2],
                   [dt     ,       0],
                   [0      ,      dt]])

    Q = np.dot(np.dot(fq, process_noise_cov), fq.T)
    return Q

def H_Jacobian_fun(x, z_dim, x_dim, dt=None):
    px, py, vx, vy = x[0, 0], x[1, 0], x[2, 0], x[3, 0]

    H = np.zeros((z_dim, x_dim))
    pxpy_squared = px**2 + py**2
    pxpy_squared_sqrt = math.sqrt(pxpy_squared)
    pxpy_pow_3over2 = pxpy_squared * pxpy_squared_sqrt
    
    H[0, 0] = px / pxpy_squared_sqrt
    H[0, 1] = py / pxpy_squared_sqrt

    H[1, 0] = -py / pxpy_squared
    H[1, 1] = px / pxpy_squared

    H[2, 0] = (py * (vx*py - vy*px)) / pxpy_pow_3over2
    H[2, 1] = (px * (vy*px - vx*py)) / pxpy_pow_3over2
    H[2, 2] = px / pxpy_squared_sqrt
    H[2, 3] = py / pxpy_squared_sqrt

    return H

def H_fun(x, dt=None):
    px, py, vx, vy = x[0, 0], x[1, 0], x[2, 0], x[3, 0]

    rho = math.sqrt(px**2 + py**2)
    if(math.fabs(rho) < 0.0001):
        print("ProcessMeasurement () - Error - Division by Zero")
        phi = math.atan2(py/px)
        ro_dot = 0
    else:
        phi = math.atan2(py, px)
        ro_dot = (px*vx+py*vy) / rho
    
    return np.array([rho, phi, ro_dot]).reshape(3,1)


if __name__ == '__main__':
    z_dim = 3
    x_dim = 4
    
    ax_noise_std = 9
    ay_noise_std = 9

    ro_noise_std = 0.9
    si_noise_std = 0.009
    ro_dot_noise_std = 0.9

    process_noise_covariance = np.array([[ax_noise_std**2,               0],
                                         [0              , ay_noise_std**2]])
    measument_noise_covariance = np.array([[ro_noise_std**2,               0,                   0],
                                           [              0, si_noise_std**2,                   0],
                                           [              0,               0, ro_dot_noise_std**2]]) 

    EKF_tracker = ExtendedKalmanFilter(x_dim, z_dim, F_fun=F_fun, H_fun=H_fun, 
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

    RMSE = EKF_tracker.calculate_rmse(estimations, radar_groundtruth).T
    print(RMSE)

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
    plt.savefig('./assert/EKF_CV_radar_Innovation.png')
    # plt.show()

    fig = plt.figure(figsize = [8,8])
    plt.scatter(radar_groundtruth[:, 0], radar_groundtruth[:, 1], label='Ground Truth Measurment')
    plt.scatter(estimations[:, 0], estimations[:, 1], label='Kalman Filter Estimation')
    plt.title('Ground Truth Vs EKF Estimation on LiDAR Data')
    plt.xlabel('X (position)')
    plt.ylabel('Y (Position)')
    plt.legend()
    plt.savefig('./assert/EKF_CV_radar.png')
    plt.show()