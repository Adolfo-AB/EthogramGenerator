import time
import data as Data
import event as Event
import eventfinder as EventFinder
from matplotlib import pyplot as plt


start_time = time.time()

### Initialize EventFinder with sigma and w
sigma = 6
w = 20
finder = EventFinder.EventFinder(sigma, w)

### Define the names of the datasets that we will use
filenames = ['7501394_PHAAET_rec16112018_PRincon_S1']

### Detect events for a given datasets
for filename in filenames:
    
    path ='C:\\Users\\adolf\\Documents\\Adolfo\\TFG\\Data\\Accelerometria\\Rabijunco\\'+filename+'\\'
    computing_time = time.time()
    
    # Load data and filter acceleration signals with a butterworth filter
    data = Data.Data(filename, path)
    data.filter_accelerations(4, 0.4)
    print("Data loaded: "+filename)
    
    ### Find raw events for ax, ay and az.
    events_ax = finder.FindEvents(filename, data.ax, "x")
    events_ay = finder.FindEvents(filename, data.ay, "y")
    events_az = finder.FindEvents(filename, data.az, "z")
    
    ### Put the events from the three axis into the same list.
    initial_events = events_ax + events_ay + events_az
    print("Initial events found: "+str(len(initial_events)))

    ### Find overlapping events    
    
    
#print("Total number of events: "+str(len(events)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")