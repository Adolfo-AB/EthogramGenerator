import csv
import numpy as np
import pandas as pd
from scipy import signal

### Data model for acceleration data.
class Data():
    def __init__(self, filename, ax, ay, az):
        
        self.filename = filename
        self.ax = ax
        self.ay = ay
        self.az = az
        
        self.pressure = None
        
    ### Filter acceleration signals using a Butterworth filter.   
    def filter_accelerations(self, N, Wn):
        b, a = signal.butter(N, Wn)
        
        self.ax = signal.filtfilt(b, a, self.ax)
        self.ay = signal.filtfilt(b, a, self.ay)
        self.az = signal.filtfilt(b, a, self.az)
        
        
        
