import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

if __name__ == '__main__':
    x_dim = 3
    z_dim = 1

    dt = 0.1
    A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])

    q_delta = np.array([0, 0, 1]).reshape(3, 1)
    q_c = np.array([0.5]).reshape(1,1)
    Q = np.dot(np.dot(q_delta, q_c), q_delta.T)
    
    H = np.array([1, 0, 0]).reshape(1, 3) 
    R = np.array([0.5]).reshape(1, 1)
    
    kf = KalmanFilter(x_dim, z_dim, A=A, H=H, Q=Q, R=R)

    x = np.linspace(-10, 10, 200)
    measurements = x**2 + 2*x - 2  + np.random.normal(0, 2, 200)
    
    predictions = []
    for z in measurements:
        kf.predict()
        pred_measurment = kf.update(z)
        predictions.append(pred_measurment[0])

    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.savefig('./assert/kf_out.png')
    plt.show()