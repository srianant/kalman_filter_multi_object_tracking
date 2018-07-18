'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from common import dprint
from scipy.optimize import linear_sum_assignment
from cv2 import KalmanFilter


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, detection, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = self.init_kalmanfilter(detection)  # KF instance to track this object

        self.trace = []  # trace path

    @staticmethod
    def init_kalmanfilter(detection):
        KF = KalmanFilter(4, 2)
        dt = 1.0 / 30.0
        KF.transitionMatrix = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]).astype(np.float32)
        KF.processNoiseCov = (np.diag([2, 2, 20, 20]).astype(np.float32) ** 2) * dt

        KF.statePost = np.array([[detection[0]],
                                 [detection[1]],
                                 [0],
                                 [0]]).astype(np.float32)
        KF.errorCovPost = np.diag([3, 3, 40, 40]).astype(np.float32) ** 2

        KF.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]]).astype(np.float32)
        KF.measurementNoiseCov = (np.eye(2).astype(np.float32) * 1) ** 2

        KF.statePre = KF.statePost  # maybe necessary?
        KF.errorCovPre = KF.errorCovPost
        return KF

    def predict(self):
        x = self.KF.predict()
        y = np.dot(self.KF.measurementMatrix, x)
        return y

    def correct(self, y):
        self.KF.correct(np.array(y).astype(np.float32))
        y = np.dot(self.KF.measurementMatrix, self.KF.statePost)
        return y

    def position_error(self):
        return np.sqrt(self.KF.errorCovPost[0, 0] + self.KF.errorCovPost[1, 1])

    def position(self):
        y = np.dot(self.KF.measurementMatrix, self.KF.statePost)
        return(y)


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_position_error, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_position_error = max_position_error
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                diff = self.tracks[i].position() - detections[j]
                distance = np.sqrt(diff[0][0]**2 +
                                   diff[1][0]**2)
                cost[i][j] = distance

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1 and cost[i][assignment[i]] > self.dist_thresh):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                assignment[i] = -1
                un_assigned_tracks.append(i)

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].position_error() > self.max_position_error):
                del_tracks.append(i)

        for id in np.flipud(del_tracks):  # only when skipped frame exceeds max
            if id < len(self.tracks):
                del self.tracks[id]
                del assignment[id]
            else:
                dprint("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        for id in un_assigned_detects:
            track = Track(detections[id], self.trackIdCount)
            self.trackIdCount += 1
            self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].predict()

            if(assignment[i] != -1):
                self.tracks[i].correct(detections[assignment[i]])

            if(len(self.tracks[i].trace) > self.max_trace_length):
                del self.tracks[i].trace[0]

            self.tracks[i].trace.append(self.tracks[i].position())
