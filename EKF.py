import math
import numpy as np

class ExtendedKalmanFilter(object):
    """
    This class implements the Extended kalman filter quations.
    
    arguments:
        dim_x: integer
            Dimension of the state vector.
        dim_z: integer 
            Dimension of the measurement vector.
        x : np.array(dim_x, 1)
            Current state vector.
        P : np.array(dim_x, dim_x)
            Current state vector covariance.
        F_ : np.array(dim_x, dim_x)
            Transition matrix.
        F_fun : Function
            In linear case, it update the F_ matrix.
            In non-linear case, it calculates f(x).
        F_Jacobian_fun : 
            Jacobbian function of the F.
        Q : np.array(dim_x, dim_x)
            Process noise covariance.
        H_ : np.array(z_dim, x_dim)
            Measurment ransition matrix.
        H_fun : Function
            In linear case, it update the H_ matrix.
            In non-linear case, it calculates H(x).
        H_Jacobian_fun : Function
            Jacobbian function of the H.
        R : np.array(dim_z, dim_z)
            Measurement nosie covariance.
    """
    def __init__(self, x_dim, z_dim, F_fun, F_Jacobian_fun=None, H_fun = None, H_Jacobian_fun = None, 
                       Q_update_fun = None, process_noise_cov=None, measument_noise_cov=None):

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.x = np.ones((self.x_dim, 1))
        self.P = np.eye(self.x_dim) 

        self.F_fun = F_fun
        self.F_Jacobian_fun = F_Jacobian_fun
        self.F_ = np.ones((x_dim, x_dim)) 

        self.H_fun = H_fun
        self.H_Jacobian_fun = H_Jacobian_fun
        self.H_ = np.ones((z_dim, x_dim))

        self.Q = np.ones((self.x_dim, self.x_dim))
        self.Q_update_fun = Q_update_fun
        self.process_noise_cov = process_noise_cov
        self.R = measument_noise_cov

        self.prev_time_stamp = 0
        self.dt = 0
    
    def predict(self):
        """
        This function to perform the prediction step in kalman filter, to compute 
        the predict density by calculating the mean and the covariance of the state
        """

        self.Q = self.Q_update_fun(self.x, self.process_noise_cov, self.dt)

        if self.F_Jacobian_fun is None:
            self.F_ = self.F_fun(self.x, self.dt)
            self.x  = np.dot(self.F_, self.x)
            self.P  = np.dot((np.dot(self.F_, self.P)), self.F_.T) + self.Q
        else:
            self.F_ = self.F_Jacobian_fun(self.x, self.z_dim, self.x_dim, self.dt)
            self.x = self.F_fun(self.x, self.dt)
            self.P  = np.dot((np.dot(self.F_, self.P)), self.F_.T) + self.Q

    def update(self, measurment):
        """
        This function to perform the update step in the kalman filter, to compute the
        posterior density, given the prior density. it compute the kalman gain (K), 
        then the innivation gain (V), and the innovation covariance (S). Finally it return 
        the mean and covariance of the posterior density.
        """
        if self.H_Jacobian_fun is None:
            self.H_ = self.H_fun(self.x, self.dt)
            predicted_measurent = np.dot(self.H_, self.x)
        else:
            self.H_ = self.H_Jacobian_fun(self.x, self.z_dim, self.x_dim, self.dt)
            predicted_measurent = self.H_fun(self.x)

        V = measurment - predicted_measurent
        S = np.dot(np.dot(self.H_, self.P), self.H_.T) + self.R 
        K = np.dot(np.dot(self.P, self.H_.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, V)
        self.P = self.P - np.dot(np.dot(K, S), K.T)
        
        return V

    def calculate_rmse(self, estimations, ground_truth):
        '''
        Root Mean Squared Error.
        '''
        if len(estimations) != len(ground_truth) or len(estimations) == 0:
            raise ValueError('calculate_rmse () - Error - estimations and ground_truth must match in length.')

        rmse = np.zeros((self.x_dim, 1))

        for est, gt in zip(estimations, ground_truth):
            rmse += np.square(est - gt)

        rmse /= len(estimations)
        return np.sqrt(rmse)