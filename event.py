
class Event():
    def __init__(self, start, end, axis, filename): 
        
        self.start = start #Sample at which the detected event starts.
        self.end = end #Sample at which the detected event finishes.
        self.axis = axis #Axis where the event was detected.
        self.filename = filename #Filename where the event was detected.
        
        self.ax = None
        self.ay = None
        self.az = None
        self.pressure = None
        
    def setup_acceleration(self, data):
        self.ax = data.ax[self.start:self.end]
        self.ay = data.ay[self.start:self.end]
        self.az = data.az[self.start:self.end]
        