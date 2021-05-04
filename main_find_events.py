#%% Cell 1: Find events for given datasets.
import time
import copy
import data as Data
import event as Event
import eventfinder as EventFinder

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
    events = copy.copy(initial_events)
    events = finder.FindOverlappingEvents(filename, events, len(data.ax))
    print("Events found after overlapping: "+str(len(events)))
    
    ### Add acceleration data to the events.
    for event in events:
        event.setup_acceleration(data)
        
    ### Run some tests to ensure that the code has worked as expected.
    number_of_errors = finder.TestCheckTagCoherence(events, data)
    if number_of_errors > 0:
        print("Some of the events do not have the right axis label assigned. Number of errors: "+str(number_of_errors))
    
    number_of_errors = finder.TestCheckEveryInitialEventIsInsideAFinalEvent(initial_events, events)
    if number_of_errors > 0:
        print("Some of the initial events is not inside a final event. Number of errors: "+str(number_of_errors))
        
print("Total number of events: "+str(len(events)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 2: Plot events given a condition.
from matplotlib import pyplot as plt

for event in events:
    if len(event.axis) == 3:
        event.setup_acceleration(data)
        
        fig, ax = plt.subplots(3,1,figsize = (8,6))
        ax[0].title.set_text('Event Axis: '+event.axis)
        ax[0].plot(event.ax, 'b-')
        ax[0].set_ylim([-9, 9])
        ax[1].plot(event.ay, 'g-')
        ax[1].set_ylim([-9, 9])
        ax[2].plot(event.az, 'r-')
        ax[2].set_ylim([-9, 9])
        plt.show()