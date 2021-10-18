import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from kalman_filter import KalmanFilter

def Update_A(dt):
        '''
        updates the motion model and process covar based on delta time from last measurement.
        '''
        A   = np.array([[1,0,dt,0],
                        [0,1,0,dt],
                        [0,0,1,0],
                        [0,0,0,1]])
        return A

if __name__ == '__main__': 
    ################################## Define Kalman Filter Parameters ##################################
    x_dim = 4
    z_dim = 2

    dt = 0.0
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])

    # q_delta: np.array(x_dim, z_dim)
    q_delta = np.array([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]])
    # q_c: np.array(z_dim, z_dim)
    q_c = np.array([[1, 0],
                    [0, 1]])
    # Q: np.array(x_dim, x_dim)
    Q = np.dot(np.dot(q_delta, q_c), q_delta.T)
    # H = np.array(z_dim, x_dim)
    H = np.array([[1,0,0,0],
                  [0,1,0,0]])
    R = np.eye(z_dim)
    
    kf = KalmanFilter(x_dim, z_dim, A=A, H=H, Q=Q, R=R)

    ################################## Read the sensor Measurments ##################################
    M = utils.Measurments('./data/sample-laser-radar-measurement-data-1.txt')
    lidar_measurments = M.lidar_measurments
    lidar_groundtruth = M.lidar_ground_truth
    
    ################################## Run KF Tracker ###############################################
    predictions = []
    estimations = []
    for z in lidar_measurments:
        current_time_stamp = z[2]
        z = z[:2].reshape(z_dim, 1)

        dt =( current_time_stamp - kf.previous_time_stamp) / 1000000.0
        kf.previous_time_stamp =current_time_stamp
        kf.A = Update_A(dt)

        kf.predict()
        pred_measurment = kf.update(z)

        predictions.append(pred_measurment)
        estimations.append(kf.x)

    ################################## Evaluate the performance ##################################
    lidar_groundtruth = np.array(lidar_groundtruth)
    predictions = np.array(predictions)
    estimations = np.array(estimations)
    lidar_groundtruth = np.expand_dims(lidar_groundtruth, axis=2)
    RMSE = kf.calculate_rmse(estimations, lidar_groundtruth).T
    print(RMSE)

    plt.scatter(lidar_groundtruth[:, 0], lidar_groundtruth[:, 1], label='Ground Truth Measurment')
    plt.scatter(predictions[:, 0], predictions[:, 1], label='Kalman Filter Estimation')
    plt.title('Ground Truth Vs KF Estimation on LiDAR Data')
    plt.xlabel('X (position)')
    plt.ylabel('Y (Position)')
    plt.legend()
    plt.savefig('./assert/kf_lidar_data.png')
    plt.show()