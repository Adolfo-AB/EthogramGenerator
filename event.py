
class Event():
    def __init__(self, start, end, axis, filename): 
        
        self.start = start #Sample at which the detected event starts.
        self.end = end #Sample at which the detected event finishes.
        self.axis = axis #Axis where the event was detected.
        self.filename = filename #Filename where the event was detected.
        
        self.id = None
        self.ax = None
        self.ay = None
        self.az = None
        self.pressure = None
        
        self.upper_threshold_ax = None
        self.upper_threshold_ay = None
        self.upper_threshold_az = None
        
        self.lower_threshold_ax = None
        self.lower_threshold_ay = None
        self.lower_threshold_az = None
        
        self.group_label = None
        
    def setup_acceleration(self, data):
        self.ax = data.ax[int(float(self.start)):int(float(self.end))]
        self.ay = data.ay[int(float(self.start)):int(float(self.end))]
        self.az = data.az[int(float(self.start)):int(float(self.end))]
        self.pressure = data.pressure[int(float(self.start)):int(float(self.end))]
        
    def setup_thresholds(self, upper_threshold_ax, lower_threshold_ax, upper_threshold_ay, lower_threshold_ay, upper_threshold_az, lower_threshold_az):
        self.upper_threshold_ax = upper_threshold_ax
        self.upper_threshold_ay = upper_threshold_ay
        self.upper_threshold_az = upper_threshold_az
        
        self.lower_threshold_ax = lower_threshold_ax
        self.lower_threshold_ay = lower_threshold_ay
        self.lower_threshold_az = lower_threshold_az