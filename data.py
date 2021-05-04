import csv
import numpy as np
import pandas as pd
from scipy import signal

class Data():
    def __init__(self, filename, path):
        
        self.filename = filename
        self.path = path
        
        try:
            data = pd.read_csv(path + filename + '.csv', header=0, sep=';')
            accelerations = data.loc[:, ['X', 'Y', 'Z']].values
        except:
            data = pd.read_csv(path + filename + '.csv', header=0)
            accelerations = data.loc[:, ['X', 'Y', 'Z']].values  
        
        self.ax = accelerations[:,0]
        self.ay = accelerations[:,1]
        self.az = accelerations[:,2]
        self.pressure = data.loc[:, ['Pressure']].values 
        
        
    def filter_accelerations(self, N, Wn):
        b, a = signal.butter(N, Wn)
        
        self.ax = signal.filtfilt(b, a, self.ax)
        self.ay = signal.filtfilt(b, a, self.ay)
        self.az = signal.filtfilt(b, a, self.az)
        
        
        
        
