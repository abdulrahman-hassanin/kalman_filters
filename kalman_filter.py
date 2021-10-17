import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter():
    """
    This class implements the kalman filter quations.
    
    arguments:
        dim_x: integer
            Dimension of the state vector.
        dim_z: integer 
            Dimension of the measurement vector.
        x : np.array(dim_x, 1)
            Current state vector mean.
        P : np.array(dim_x, dim_x)
            Current state vector covariance. 
        A : np.array(dim_x, dim_x)
            State transition matrix.
        Q : np.array(dim_x, dim_x)
            Process noise covariance.
        H : np.array(dim_z, dim_x)
            Measurement transition matrix.
        R : np.array(dim_z, dim_z)
            Measurement nosie covariance.
    """
    def __init__(self, dim_x, dim_z, x=None, P=None, A=None, H=None, Q=None, R=None):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((self.dim_x, 1)) if x is None else x
        self.P = np.eye(self.dim_x) if P is None else P
        self.A = np.eye(self.dim_x) if A is None else A
        self.Q = np.eye(self.dim_x) if Q is None else Q
        self.H = np.zeros((self.dim_z, self.dim_x)) if H is None else H
        self.R = np.eye(self.dim_z) if R is None else R

    def predict(self):
        """
        This function to perform the prediction step in kalman filter, to compute 
        the predict density by calculating the mean and the covariance of the state
        """
        self.x = np.dot(self.A, self.x)
        self.P = np.dot((np.dot(self.A, self.P)), self.A.T) + self.Q
        

    def update(self, measurment):
        """
        This function to perform the update step in the kalman filter, to compute the
        posterior density, given the prior density. it compute the kalman gain (K), 
        then the innivation gain (V), and the innovation covariance (S). Finally it return 
        the mean and covariance of the posterior density.
        """
        predicted_measurent = np.dot(self.H, self.x)
        V = measurment - predicted_measurent
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R 
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, V)
        self.P = self.P - np.dot(np.dot(K, S), K.T)
        return predicted_measurent

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

    x = np.linspace(-10, 10, 100)
    measurements = x**2 + 2*x - 2  + np.random.normal(0, 2, 100)
    
    predictions = []
    for z in measurements:
        kf.predict()
        pred_measurment = kf.update(z)
        predictions.append(pred_measurment[0])

    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend()
    # plt.savefig('./assert/kf_out.png')
    plt.show()