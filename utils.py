import numpy as np
import pandas as pd

class Measurments(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_packets = None
        self.lidar_measurments = []
        self.lidar_ground_truth = []
        self.radar_measurments = []
        self.radar_ground_truth = []

        self.get_data()
    
    def get_data(self):
        self.data_packets = pd.read_csv(self.data_path, header=None, sep='\t', names=['x'+str(x) for x in range(9)])
        for _ , raw_measurement_packet in self.data_packets.iterrows():  
            if raw_measurement_packet[0] == 'R':
                self.update_radar_packets(raw_measurement_packet)
            else:
                self.update_lidar_packets(raw_measurement_packet)

    def update_radar_packets(self, packet):
        rho_measured    = packet[1]
        phi_measured    = packet[2]
        rhodot_measured = packet[3]
        timestamp       = packet[4]
        x_groundtruth   = packet[5]
        y_groundtruth   = packet[6]
        vx_groundtruth  = packet[7]
        vy_groundtruth  = packet[8]
        
        Measurment = np.array([rho_measured, phi_measured, rhodot_measured, timestamp])
        ground_truth = np.array([x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth])

        self.radar_measurments.append(Measurment)
        self.radar_ground_truth.append(ground_truth)

    def update_lidar_packets(self, packet):
        x_measured     = packet[1]
        y_measured     = packet[2]
        timestamp      = packet[3]
        x_groundtruth  = packet[4]
        y_groundtruth  = packet[5]
        vx_groundtruth = packet[6]
        vy_groundtruth = packet[7]
        
        Measurment = np.array([x_measured, y_measured, timestamp])
        ground_truth = np.array([x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth])

        self.lidar_measurments.append(Measurment)
        self.lidar_ground_truth.append(ground_truth)