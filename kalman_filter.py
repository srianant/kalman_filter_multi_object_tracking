'''
    File name         : kalman_filter.py
    File Description  : Kalman Filter Algorithm Implementation
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self, y):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = 1.0 / 30.0  # delta time (>> ffmpeg -i TrackingBugs.mp4)

        self.H = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]])  # matrix in observation equations
        self.x = np.append(y, [0.0, 0]).reshape(-1, 1)

        self.P = np.diag((5.0, 5.0, 100.0, 100.0))  # covariance matrix
        self.F = np.array([[1.0, 0.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0, self.dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])  # state transition mat

        self.Q = np.diag((2.0, 2.0, 5.0, 5.0))  # process noise matrix
        self.R = np.eye(y.shape[0])  # observation noise matrix

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted output estimate
        """
        # Predicted state estimate
        self.x = np.dot(self.F, self.x)
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        # self.lastResult = self.x  # same last predicted result
        y = np.dot(self.H, self.x)
        return y

    def correct(self, y):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        H: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = HP_{k|k-1} H.T + R
            K_{k} = P_{k|k-1} H.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Hu'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                H.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        # if not flag:  # update using prediction
        #     self.b = self.lastResult
        # else:  # update using detection
        #     self.b = b
        C = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(C)))

        self.x = self.x + np.dot(K, (y - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        ypost = np.dot(self.H, self.x)
        return ypost
