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
filenames = ['8201653_PHAAET_I.Cima_rec21012021_ninho 39_36_S1']

### Detect events for a given datasets
for filename in filenames:
    
    path ='D:\\AdolfoAB\\cobas_infinity_3.02\\Rabijunco\\'+filename+'\\'
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
    export_path = "D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\"
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
    
export_path = "D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\"
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
    if len(event.ax) < 2000:
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
    if len(event.axis) == 3:
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

#%% Cell 4: Compute max correlation between each event.
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

full_corr_ax.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
full_corr_ay.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\corr_ay_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
full_corr_az.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\corr_az_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
full_lag_ax.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\lag_ax_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 4.1 Export correlation matrix and lag
import numpy as np
full_corr_ax = np.array(total_maxcorr_ax)
full_corr_ay = np.array(total_maxcorr_ay)
full_corr_az = np.array(total_maxcorr_az)
full_lag_ax = np.array(total_lag_ax)

full_corr_ax.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
full_corr_ay.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\corr_ay_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
full_corr_az.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\corr_az_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
full_lag_ax.tofile('D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\lag_ax_s'+str(sigma)+'_w'+str(w)+'.csv', sep = ',')
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

#%% Cell 6: Load events, correlation matrix and lag matrix
import datamanager as DataManager
import csv

start_time = time.time()

sigma = 6
w = 50
path = "D:\\AdolfoAB\\cobas_infinity_3.02\\Output_09052021\\"
filenames = ['8201653_PHAAET_I.Cima_rec21012021_ninho 39_36_S1']
datamanager = DataManager.DataManager()

all_data = []
for filename in filenames:
    datapath ='D:\\AdolfoAB\\cobas_infinity_3.02\\Rabijunco\\'+filename+'\\'    
    # Load data and filter acceleration signals with a butterworth filter
    data = datamanager.LoadData(filename, datapath)
    data.filter_accelerations(4, 0.4)
    all_data.append(data)
    print("Data loaded: "+filename)
    
manager = DataManager.DataManager()
all_events = manager.LoadAllEvents(path, sigma, w)
for data in all_data:
    for event in all_events:
        if event.filename == data.filename:
            event.setup_acceleration(data)

full_corr_ax, full_corr_ay, full_corr_az, full_lag_ax = manager.LoadCorrelationMatrix2(path, sigma, w, len(all_events))

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 7: Group events based on max correlation
import copy
import time
import eventgrouper as EventGrouper

start_time = time.time()

grouper = EventGrouper.EventGrouper()
threshold_ax = 0.45
threshold_ay = 0.1
threshold_az = 0.2

input_events = copy.copy(all_events)
similar_events_groups = grouper.GroupSimilarEvents(input_events, full_corr_ax, full_corr_ay, full_corr_az, threshold_ax, threshold_ay, threshold_az)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 7.1: Remove the groups that have less than N elements.
import copy

event_groups = []
temp_similar_groups = copy.copy(similar_events_groups)
min_group_size = 50

for eventgroup in temp_similar_groups:
    if len(eventgroup) > min_group_size:
        event_groups.append(eventgroup)
        

#%% Cell 8: Align events from the same group.
events_to_align = copy.copy(event_groups)
similar_events_aligned = grouper.AlignSimilarEvents(events_to_align, full_lag_ax)

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
print("Mean group size: "+str(np.mean(group_sizes)))

### Compute total number of events after processing.
number_of_events = 0
number_of_groups = 0
for eventgroup in similar_events_aligned:
    number_of_groups = number_of_groups + 1
    for event in eventgroup:
        number_of_events = number_of_events + 1
print("Total number of events after processing: "+str(number_of_events))
print("Total number of groups after processing: "+str(number_of_groups))

### Compute mean correlation coefficient (only takes account groups with size > 1)
temp_similar_groups = copy.deepcopy(similar_events_aligned)
corr_coefs = []
for event_group in temp_similar_groups:
    group_event_sizes = []
    for event in event_group:
        group_event_sizes.append(len(event.ax))
    min_event_size = min(group_event_sizes)
    
    group_ax = []
    for event in event_group:
        event.end = event.start + min_event_size
        
        for eventdata in all_data:
            if event.filename == eventdata.filename:
                event.setup_acceleration(eventdata)
                
        group_ax.append(np.array(event.ax))
    corr_coefs.append(np.mean(np.corrcoef(group_ax)))
        
print("Mean correlation coefficients: "+str(np.mean(corr_coefs)))

#%% Cell 9: Plot N events from a group.
from matplotlib import pyplot as plt

def get_cmap(n, name='YlOrRd'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

groupnumber = 14
number_of_events = 10000
cmap1 = get_cmap(len(similar_events_aligned[groupnumber]))
cmap2 = get_cmap(len(similar_events_aligned[groupnumber]), "Blues")

fig, ax = plt.subplots(2,1,figsize = (8,6))
i = 0
for event in similar_events_aligned[groupnumber]:
    if i < number_of_events and len(event.ax) < 1000000:
        ax[0].title.set_text("Group number "+str(groupnumber)+'. Group size = '+str(len(similar_events_aligned[groupnumber])))
        ax[0].plot(event.ax, c=cmap1(i), lw=0.8)
        ax[0].set_ylim([-9, 9])
        ax[1].plot(event.pressure, c=cmap2(i), lw=0.8)
        ax[1].set_ylim([990,1200])
        #ax[0].plot(np.full(len(event.ax), event.upper_threshold_ax), 'b-', ls=('dotted'))
        #ax[0].plot(np.full(len(event.ax), event.lower_threshold_ax), 'b-', ls=('dotted'))
        #ax[1].plot(event.ay, c=cmap(i))
        #ax[1].set_ylim([-9, 9])
        #ax[2].plot(event.az, c=cmap(i))
        #ax[2].set_ylim([-9, 9])
        i = i+1
plt.show()

#%% Cell 10: Set group label to each event and prepare the events for the reservoir computing model.
events_processed = []
group_label = 0
for group in similar_events_aligned:
    for event in group:
        event.group_label = group_label
        events_processed.append(event)
    group_label = group_label + 1
    

#%% Cell 11: Testing reservoir
import network as Network
import data2 as Data
import scipy.io

test_name = "5s"

dd = Data.Data(80)
Network = Network.Network()

dd.import_data('D:\\AdolfoAB\\cobas_infinity_3.02\\dataSorted_allOrientations.mat')
num_nodes = 20

dd.build_train_labels_lin()
dd.build_test_labels_lin()

dd.build_training_matrix()
dd.build_test_matrix()


input_probability = 0.1
reservoir_probability = 0.1
classifier = "lin"
Network.T = dd.training_data.shape[1] #Number of training time steps
Network.n_min = 2540 #Number time steps dismissed
Network.K = 128 #Input layer size
Network.N = num_nodes #Reservoir layer size


Network.u = dd.training_data
Network.y_teach = dd.training_results

Network.setup_network(dd,num_nodes,input_probability,reservoir_probability,dd.data.shape[-1])

Network.train_network(dd.data.shape[-1],classifier,dd.num_columns, dd.num_trials_train, dd.train_labels, Network.N)

Network.mean_test_matrix = np.zeros([Network.N,dd.num_trials_test,dd.data.shape[-1]])

Network.test_network(dd.test_data, dd.num_columns,dd.num_trials_test, Network.N, dd.data.shape[-1], t_autonom=dd.test_data.shape[1])

if classifier == 'lin':
	print(f'Performance for {test_name} using {classifier} : {dd.accuracy_lin(Network.regressor.predict(Network.mean_test_matrix.T),dd.test_labels)}')

elif classifier == 'log':
	print(f'Performance for {test_name} using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,dd.test_labels.T)}')

elif classifier == '1nn':
	print(f'Performance for {test_name} using {classifier} : {Network.regressor.score(Network.mean_test_matrix.T,dd.test_labels)}')


#%% Cell 12: Create data structures needed for reservoir.
import numpy as np

num_groups = len(similar_events_aligned)
num_events = len(events_processed)

input_data_ax, input_data_ay, input_data_az = [], [], []
len_events = []
labels = np.zeros((num_events, num_groups))

for event in events_processed:
    len_events.append(len(event.ax))
    labels[events_processed.index(event), event.group_label] = 1
    for ax in event.ax:
        input_data_ax.append(ax)
    for ay in event.ay:
        input_data_ay.append(ay)
    for az in event.az:
        input_data_az.append(az)

input_data = np.array([np.array(input_data_ax), np.array(input_data_ay), np.array(input_data_az)])


#%% Cell 13: Trying reservoir with my own data.
import copy
import network2 as Network2

Network2 = Network2.Network()
num_nodes = 10

num_events_train = int(1*num_events)
input_probability = 0.5
reservoir_probability = 0.5
classifier = "lin"

Network2.T = sum(len_events)  
Network2.n_min = 2450
Network2.K = 3
Network2.N = num_nodes

Network2.setup_network(input_data, num_nodes, input_probability, reservoir_probability, num_groups, num_events_train)
Network2.train_network(num_groups, classifier, num_events, len_events, labels, num_nodes)


