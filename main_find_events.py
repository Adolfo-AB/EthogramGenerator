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
w = 50
mode = "mean"
finder = EventFinder.EventFinder(sigma, w, mode)

### Initialize a list to store the events from all the datasets.
all_data = []
all_events = []

### Define the names of the datasets that we will use
filenames = ['7501394_PHAAET_rec16112018_PRincon_S1',
             '7501709_PHAAET_rec18112018_PRincon_S1',
             '7501755_PHAAET_rec27112018_PRincon_S1', 
             '8200163_PHAAET_rec14052019_PRoque_S1',
             '8200445_PHAAET_rec290422019_PRincon_S1',
             '8200473_PHAAET_rec24052019_PRincon_S2',
             '8200487_PHAAET_rec04052019_PRincon_S1',
             '8200718_PHAAET_rec08032019_PRincon',
             '8201653_PHAAET_I.Cima_rec21012021_ninho 39_36_S1',
             '8201667_PHAAET_I.Cima_rec21012021_ninho 68_21_S1',
             '8201720_PHAAET_rec31122020_ICima_ninho 71_21_S1',
             '8201959_PHAAET_rec29122020_ICima_ninho 31_36_S1']

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
    threshold = 50
    events = finder.RemoveShortEvents(overlapping_events, threshold)
    print("Number of events after removing short evernts: "+str(len(events)))
    
    ### Add acceleration data and event id to each event.
    for event in events:
        event.id = events.index(event)
    
    ### Export events from filename to CSV
    export_path = "C:\\Users\\adolf\\TFG\\Output_08052021\\"
    datamanager.ExportEventsToCSV(events, finder.sigma, finder.w, filename, export_path)
    print("Events successfully exported to .csv.")
    print("")
    ### Append events into all_events.
    all_events = all_events + events

### Export all events to CSV
i = 0
for event in all_events:
    event.id = i
    i = i+1
    
export_path = "C:\\Users\\adolf\\TFG\\Output_08052021\\"
datamanager.ExportAllEventsToCSV(all_events, finder.sigma, finder.w, export_path)
print("All events successfully exported to .csv.")
print("")
   
print("Total number of events: "+str(len(all_events)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 2: Find some useful metrics for the detected events.
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

#%% Cell 2.1: Plot event length histogram
from matplotlib import pyplot as plt

event_lengths = []                
for event in all_events:
    if len(event.ax) < 200000:
        event_lengths.append(len(event.ax))

### Plot event length histogram
hist, bin_edges = np.histogram(event_lengths)

fig, ax = plt.subplots(1,1,figsize = (8,6))
ax.hist(event_lengths, bins=200, log=True)  # arguments are passed to np.histogram)

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
    if len(event.axis) == 2:
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

#%% Cell 4: Group events based on max correlation
import numpy as np
import eventgrouper as EventGrouper
start_time = time.time()

grouper = EventGrouper.EventGrouper()

corr_events = copy.copy(all_events)

i = 0
for event in corr_events:
    event.id = i
    i = i + 1
    
total_maxcorr_ax, total_maxcorr_ay, total_maxcorr_az, total_lag_ax, total_lag_ay, total_lag_az = grouper.ComputeMaxCorrelationMatrix(corr_events)

full_corr_ax = np.array(total_maxcorr_ax)
full_corr_ay = np.array(total_maxcorr_ay)
full_corr_az = np.array(total_maxcorr_az)
full_lag_ax = np.array(total_lag_ax)

full_corr_ax.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\corr_ax.csv', sep = ',')
full_corr_ay.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\corr_ay.csv', sep = ',')
full_corr_az.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\corr_az.csv', sep = ',')
full_lag_ax.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\lag_ax.csv', sep = ',')

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 4.1 Export correlation matrix and lag
import numpy as np
full_corr_ax = np.array(total_maxcorr_ax)
full_corr_ay = np.array(total_maxcorr_ay)
full_corr_az = np.array(total_maxcorr_az)
full_lag_ax = np.array(total_lag_ax)

full_corr_ax.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\corr_ax.csv', sep = ',')
full_corr_ay.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\corr_ay.csv', sep = ',')
full_corr_az.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\corr_az.csv', sep = ',')
full_lag_ax.tofile('C:\\Users\\adolf\\TFG\\Output_08052021\\lag_ax.csv', sep = ',')

#%% Cell 5: Plot max correlation matrix
### Print max correlation matrix
from matplotlib import pyplot as plt
import numpy as np
   
fig, ax = plt.subplots(3,1,figsize = (32,24))
ax[0].title.set_text('Max correlation for ax axis.')
ax[0].imshow(full_corr_ax)
ax[1].title.set_text('Max correlation for ay axis.')
ax[1].imshow(full_corr_ay)
ax[2].title.set_text('Max correlation for az axis.')
ax[2].imshow(full_corr_az)
plt.show()

#%% Cell 7: Load events, correlation matrix and lag matrix
import datamanager as DataManager
import csv

path = "C:\\Users\\adolf\\TFG\\Output_06052021\\"

manager = DataManager.DataManager()
all_events = manager.LoadAllEvents(path, sigma, w)

#%% Cell 7: Group events based on max correlation
import eventgrouper as EventGrouper
start_time = time.time()

threshold_ax = 0.15
threshold_ay = 0
threshold_az = 0

input_events = copy.copy(all_events)
similar_events_groups = grouper.GroupSimilarEvents(input_events, full_corr_ax, full_corr_ay, full_corr_az, threshold_ax, threshold_ay, threshold_az)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 8: Align events from the same group.
events_to_align = copy.copy(similar_events_groups)
similar_events_aligned = grouper.AlignSimilarEvents(events_to_align, total_lag_ax)

for eventdata in all_data:
    for event_group in similar_events_aligned:
        for event in event_group:
            if event.filename == eventdata.filename:
                event.setup_acceleration(eventdata)
        
#%% Cell 8.1: Compute metrics from aligned similar groups
import numpy as np
import eventgrouper as EventGrouper
grouper = EventGrouper.EventGrouper()
             
### Compute max group size
group_sizes = []
for event_group in similar_events_aligned:
    group_sizes.append(len(event_group))
print("Maximum group size: "+str(max(group_sizes)))


### Compute mean correlation coefficient (only takes account groups with size > 1)
'''
correlation_coefficients = []
for event_group in similar_events_aligned:        
    if len(event_group) > 1:
        group_corr_ax, group_corr_ay, group_corr_az, group_lag_ax, group_lag_ay, group_lag_az = grouper.ComputeMaxCorrelationMatrix(event_group)
        
        
print("Mean correlation coefficients: "+str(np.mean(correlation_coefficients)))
'''
#%% Cell 6: Plot all the events from a group.
from matplotlib import pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

groupnumber = 35
cmap = get_cmap(len(similar_events_aligned[groupnumber]))

fig, ax = plt.subplots(1,1,figsize = (8,6))
i = 0
for event in similar_events_aligned[groupnumber]:
    if i < 10 and len(event.ax) < 1000:
        ax.title.set_text("Event id: "+str(event.id)+". Event Axis: "+event.axis)
        ax.plot(event.ax, c=cmap(i))
        #ax[0].plot(np.full(len(event.ax), event.upper_threshold_ax), 'b-', ls=('dotted'))
        #ax[0].plot(np.full(len(event.ax), event.lower_threshold_ax), 'b-', ls=('dotted'))
        ax.set_ylim([-9, 9])
        #♦ax[1].plot(event.ay, c=cmap(i))
        #ax[1].plot(np.full(len(event.ay), event.upper_threshold_ay), 'g-', ls=('dotted'))
        #ax[1].plot(np.full(len(event.ay), event.lower_threshold_ay), 'g-', ls=('dotted'))
        #ax[1].set_ylim([-9, 9])
        #ax[2].plot(event.az, c=cmap(i))
        #ax[2].plot(np.full(len(event.az), event.upper_threshold_az), 'r-', ls=('dotted'))
        #ax[2].plot(np.full(len(event.az), event.lower_threshold_az), 'r-', ls=('dotted'))
        #ax[2].set_ylim([-9, 9])
        i = i+1
plt.show()


