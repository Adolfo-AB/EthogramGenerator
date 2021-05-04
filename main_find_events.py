#%% Cell 1: Find events for given datasets.
import time
import copy
import warnings
import data as Data
import event as Event
import datamanager as DataManager
import eventfinder as EventFinder
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize DataManager class.
datamanager = DataManager.DataManager()

### Initialize EventFinder with sigma, w and mode (rest or mean).
sigma = 6
w = 12
mode = "rest"
finder = EventFinder.EventFinder(sigma, w, mode)

### Initialize a list to store the events from all the datasets.
all_data = []
all_events = []

### Define the names of the datasets that we will use
filenames = ['7501394_PHAAET_rec16112018_PRincon_S1']

### Detect events for a given datasets
for filename in filenames:
    
    path ='C:\\Users\\adolf\\Documents\\Adolfo\\TFG\\Data\\Accelerometria\\Rabijunco\\'+filename+'\\'
    computing_time = time.time()
    
    # Load data and filter acceleration signals with a butterworth filter
    data = datamanager.LoadData(filename, path)
    data.filter_accelerations(4, 0.4)
    all_data.append(data)
    print("Data loaded: "+filename)
    
    ### Find raw events for ax, ay and az.
    events_ax = finder.FindEvents(filename, data.ax, "x", "mean")
    events_ay = finder.FindEvents(filename, data.ay, "y", "mean")
    events_az = finder.FindEvents(filename, data.az, "z", "mean")
    
    ### Put the events from the three axis into the same list.
    initial_events = events_ax + events_ay + events_az
    print("Initial events found: "+str(len(initial_events)))

    ### Find overlapping events
    overlapping_events = copy.copy(initial_events)
    overlapping_events = finder.FindOverlappingEvents(filename, overlapping_events, len(data.ax))
    print("Events found after overlapping: "+str(len(overlapping_events)))

    ### Add acceleration data and event id to each event.
    for event in overlapping_events:
        event_data = all_data[filenames.index(event.filename)]
        event.setup_acceleration(event_data)
        event.id = overlapping_events.index(event)
        
    '''
    ### Run some tests to ensure that the code has worked as expected.
    number_of_errors = finder.TestCheckTagCoherence(overlapping_events, data)
    if number_of_errors > 0:
        print("Some of the events do not have the right axis label assigned. Number of errors: "+str(number_of_errors))
    
    number_of_errors = finder.TestCheckEveryInitialEventIsInsideAFinalEvent(initial_events, overlapping_events)
    if number_of_errors > 0:
        print("Some of the initial events is not inside a final event. Number of errors: "+str(number_of_errors))
    '''
    
    ### Remove events shorter than threshold.
    threshold = 25
    events = finder.RemoveShortEvents(overlapping_events, threshold)
    print("Number of events after removing short events: "+str(len(events)))
    
    ### Add acceleration data and event id to each event.
    for event in events:
        event.id = events.index(event)
    
    ### Export events from filename to CSV
    export_path = "C:\\Users\\adolf\\TFG\\Data_04052021\\"
    datamanager.ExportEventsToCSV(events, finder.sigma, finder.w, filename, export_path)
    print("Events successfully exported to .csv.")
    print("")
    ### Append events into all_events.
    all_events = all_events + events
    
print("Total number of events: "+str(len(all_events)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell2: Find some useful metrics for the detected events.
import numpy as np

length_events_1axis = []
length_events_2axis = []
length_events_3axis = []
for event in all_events:
    if len(event.axis) == 1:
        length_events_1axis.append(len(event.ax))
    if len(event.axis) == 2:
        length_events_2axis.append(len(event.ax))
    if len(event.axis) == 3:
        length_events_3axis.append(len(event.ax))

print("Number of 1-axis events: "+str(len(length_events_1axis)))    
print("Max 1-axis event length: "+str(max(length_events_1axis)))
print("Min 1-axis event length: "+str(min(length_events_1axis)))
print("Mean 1-axis event length: "+str(np.mean(length_events_1axis)))
print("")

print("Number of 2-axis events: "+str(len(length_events_2axis)))
print("Max 2-axis event length: "+str(max(length_events_2axis)))
print("Min 2-axis event length: "+str(min(length_events_2axis)))
print("Mean 2-axis event length: "+str(np.mean(length_events_2axis)))
print("")

print("Number of 3-axis events: "+str(len(length_events_3axis)))
print("Max 3-axis event length: "+str(max(length_events_3axis)))
print("Min 3-axis event length: "+str(min(length_events_3axis)))
print("Mean 3-axis event length: "+str(np.mean(length_events_3axis)))

#%% Cell 3: Plot events that fulfill a given condition.
from matplotlib import pyplot as plt
import numpy as np

for eventdata in all_data:
    
    std_ax = np.std(eventdata.ax)
    std_ay = np.std(eventdata.ay)
    std_az = np.std(eventdata.az)
    
    if finder.mode == "mean":
        mean_ax = np.mean(eventdata.ax)
        mean_ay = np.mean(eventdata.ay)
        mean_az = np.mean(eventdata.az)
        
        upper_threshold_ax = mean_ax + finder.sigma*std_ax
        lower_threshold_ax = mean_ax - finder.sigma*std_ax
        upper_threshold_ay = mean_ay + finder.sigma*std_ay
        lower_threshold_ay = mean_ay - finder.sigma*std_ay
        upper_threshold_az = mean_az + finder.sigma*std_az
        lower_threshold_az = mean_az - finder.sigma*std_az
        
    if finder.mode == "rest":
        upper_threshold_ax = 0 + finder.sigma*std_ax
        lower_threshold_ax = 0 - finder.sigma*std_ax
        upper_threshold_ay = 0 + finder.sigma*std_ay
        lower_threshold_ay = 0 - finder.sigma*std_ay
        upper_threshold_az = 0 + finder.sigma*std_az
        lower_threshold_az = 0 - finder.sigma*std_az
        
    for event in all_events:
        if event.filename == eventdata.filename:
            event.setup_thresholds(upper_threshold_ax, lower_threshold_ax, upper_threshold_ay, lower_threshold_ay, upper_threshold_az, lower_threshold_az)
            
for event in all_events:
    if len(event.axis) == 3:
    #if len(event.ax) > 250:
    #if event.id == 245 or event.id == 3:
        fig, ax = plt.subplots(3,1,figsize = (8,6))
        ax[0].title.set_text("Event id: "+str(event.id)+". Event Axis: "+event.axis)
        ax[0].plot(event.ax, 'b-')
        ax[0].plot(np.full(len(event.ax), event.upper_threshold_ax), 'b-', ls=('dotted'))
        ax[0].plot(np.full(len(event.ax), event.lower_threshold_ax), 'b-', ls=('dotted'))
        ax[0].set_ylim([-9, 9])
        ax[1].plot(event.ay, 'g-')
        ax[1].plot(np.full(len(event.ay), event.upper_threshold_ay), 'g-', ls=('dotted'))
        ax[1].plot(np.full(len(event.ay), event.lower_threshold_ay), 'g-', ls=('dotted'))
        ax[1].set_ylim([-9, 9])
        ax[2].plot(event.az, 'r-')
        ax[2].plot(np.full(len(event.az), event.upper_threshold_az), 'r-', ls=('dotted'))
        ax[2].plot(np.full(len(event.az), event.lower_threshold_az), 'r-', ls=('dotted'))
        ax[2].set_ylim([-9, 9])
        plt.show()