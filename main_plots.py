#%% Cell 1: Load data, filter signals and find segments.
import time
import copy
import warnings
import numpy as np
import data_manager
import segment_manager
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_managerager class.

### Initialize with sigma, w and mode (rest or mean).
sigma = 6
w = 60
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()
### Initialize a list to store the events from all the datasets.
all_data = []
all_segments = []

### Define the names of the datasets that we will use
filenames = ['8200487_PHAAET_rec04052019_PRincon_S1']

### Detect events for a given datasets
for filename in filenames:
    
    path ='D:\\AdolfoAB\\cobas_infinity_3.02\\Rabijunco\\'+filename+'\\'
    #path = 'C:\\Users\\adolf\\Documents\\Adolfo\\TFG\\Data\\Accelerometria\\Rabijunco\\'+filename+'\\'
    
    # Load data and filter acceleration signals with a butterworth filter
    initial_data = data_manager.load_data_datetimes(filename, path)
    current_data = copy.deepcopy(initial_data)
    current_data.filter_accelerations(4, 0.4)
    all_data.append(current_data)
    print("Data loaded: "+filename)
    
    '''
    ###############################
    ### Plot raw vs filtered signal
    fig, ax = plt.subplots(3,2,figsize = (8,6))
    fig.suptitle("Raw vs filtered acceleration signals")
    ax[0,0].plot(initial_data.ax[10000:10200], color='lightseagreen')
    ax[0,0].set_ylim([-3, 3])
    ax[0,1].plot(current_data.ax[10000:10200], color='lightseagreen')
    ax[0,1].set_ylim([-3, 3])
    ax[0,0].set_ylabel("ax")
    ax[1,0].plot(initial_data.ay[10000:10200], color='coral')
    ax[1,0].set_ylim([-3, 3])
    ax[1,1].plot(current_data.ay[10000:10200], color='coral')
    ax[1,1].set_ylim([-3, 3])
    ax[1,0].set_ylabel("ay")
    ax[2,0].plot(initial_data.az[10000:10200], color='olive')
    ax[2,0].set_ylim([-3, 3])
    ax[2,1].plot(current_data.az[10000:10200], color='olive')
    ax[2,1].set_ylim([-3, 3])
    ax[2,0].set_ylabel("az")
    ax[2,0].set_xlabel("Number of samples")
    ax[2,1].set_xlabel("Number of samples")
    plt.show()
    ###############################
    '''
    
    ### Find raw events for ax, ay and az.
    segments_ax = segment_manager.create_raw_segments(filename, current_data.ax, "x", "mean")
    segments_ay = segment_manager.create_raw_segments(filename, current_data.ay, "y", "mean")
    segments_az = segment_manager.create_raw_segments(filename, current_data.az, "z", "mean")
    
    ### Save initial segments into a different list to check that none of them are lost after overlapping.
    init_segments = segments_ax + segments_ay + segments_az
    print("Initial segments found: "+str(len(init_segments)))

    ### Find overlapping segments
    current_segments = copy.deepcopy(init_segments)
    current_segments = segment_manager.overlap_segments(filename, current_segments, len(current_data.ax))
    print("Segments found after overlapping: "+str(len(current_segments)))

    ### Add acceleration data and segment id to each segment.
    for segment in current_segments:
        segment.setup_acceleration(current_data)
        segment.id = current_segments.index(segment)
   
    '''
    ### Run some tests to ensure that the code has worked as expected.
    number_of_errors = segment_manager.test_tag_coherence(current_segments, current_data)
    if number_of_errors > 0:
        print("Some of the segments do not have the right axis label assigned. Number of errors: "+str(number_of_errors))
    
    number_of_errors = segment_manager.test_no_segments_missing(init_segments, current_segments)
    if number_of_errors > 0:
        print("Some of the initial segments is not inside a final segment. Number of errors: "+str(number_of_errors))
    '''
    
    ### Add segment id to each segment.
    for segment in current_segments:
        segment.id = current_segments.index(segment)
    
    ### Append the segments found in the current dataset into a list that contains the segments from ALL datasets.
    all_segments = all_segments + current_segments
    
    '''
    ##############################################
    ### Plot original signals vs segmented signals
    min_ax, max_ax = min(current_data.ax)-0.5, max(current_data.ax)+0.5
    min_ay, max_ay = min(current_data.ay)-0.5, max(current_data.ay)+0.5
    min_az, max_az = min(current_data.az)-0.5, max(current_data.az)+0.5
    
    std_ax = np.std(current_data.ax)
    std_ay = np.std(current_data.ay)
    std_az = np.std(current_data.az)
    
    if segment_manager.mode == "mean":
        mean_ax = np.mean(current_data.ax)
        mean_ay = np.mean(current_data.ay)
        mean_az = np.mean(current_data.az)
        
    upper_threshold_ax = mean_ax + segment_manager.sigma*std_ax
    lower_threshold_ax = mean_ax - segment_manager.sigma*std_ax
    upper_threshold_ay = mean_ay + segment_manager.sigma*std_ay
    lower_threshold_ay = mean_ay - segment_manager.sigma*std_ay
    upper_threshold_az = mean_az + segment_manager.sigma*std_az
    lower_threshold_az = mean_az - segment_manager.sigma*std_az
    
    fig, ax = plt.subplots(3,2,figsize = (32,24))
    fig.suptitle("Original signal vs segmented signal", fontsize = 50)
    ax[0,0].plot(current_data.timestamp, current_data.ax, color='lightseagreen')
    ax[0,0].plot(current_data.timestamp, np.full(len(current_data.ax), upper_threshold_ax), color='lightseagreen', ls=('dotted'))
    ax[0,0].plot(current_data.timestamp, np.full(len(current_data.ax), lower_threshold_ax), color='lightseagreen', ls=('dotted'))
    ax[0,0].get_xaxis().set_major_locator(mdates.HourLocator(interval=6))
    ax[0,0].get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0,0].get_xaxis().label.set_fontsize(10)
    ax[0,1].plot(current_data.timestamp, np.full(len(current_data.ax), upper_threshold_ax), color='lightseagreen', ls=('dotted'))
    ax[0,1].plot(current_data.timestamp, np.full(len(current_data.ax), lower_threshold_ax), color='lightseagreen', ls=('dotted'))
    ax[0,1].get_xaxis().set_major_locator(mdates.HourLocator(interval=6))
    ax[0,1].get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0,1].get_xaxis().label.set_fontsize(10)
    ax[0,0].set_ylim([min_ax, max_ax])
    ax[0,0].set_ylabel("acceleration (X axis)",fontsize=30)
    ax[1,0].plot(current_data.timestamp, current_data.ay, color='coral')
    ax[1,0].set_ylim([min_ay, max_ay])
    ax[1,0].plot(current_data.timestamp, np.full(len(current_data.ay), upper_threshold_ay), color='coral', ls=('dotted'))
    ax[1,0].plot(current_data.timestamp, np.full(len(current_data.ay), lower_threshold_ay), color='coral', ls=('dotted'))
    ax[1,0].get_xaxis().set_major_locator(mdates.HourLocator(interval=6))
    ax[1,0].get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[1,0].get_xaxis().label.set_fontsize(10)
    ax[1,1].plot(current_data.timestamp, np.full(len(current_data.ay), upper_threshold_ay), color='coral', ls=('dotted'))
    ax[1,1].plot(current_data.timestamp, np.full(len(current_data.ay), lower_threshold_ay), color='coral', ls=('dotted'))
    ax[1,1].get_xaxis().set_major_locator(mdates.HourLocator(interval=6))
    ax[1,1].get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[1,1].get_xaxis().label.set_fontsize(10)
    ax[1,0].set_ylabel("acceleration (Y axis)",fontsize=30)
    ax[2,0].plot(current_data.timestamp, current_data.az, color='olive')
    ax[2,0].plot(current_data.timestamp, np.full(len(current_data.az), upper_threshold_az), color='olive', ls=('dotted'))
    ax[2,0].plot(current_data.timestamp, np.full(len(current_data.az), lower_threshold_az), color='olive', ls=('dotted'))
    ax[2,0].get_xaxis().set_major_locator(mdates.HourLocator(interval=6))
    ax[2,0].get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[2,0].get_xaxis().label.set_fontsize(10)
    ax[2,1].plot(current_data.timestamp, np.full(len(current_data.az), upper_threshold_az), color='olive', ls=('dotted'))
    ax[2,1].plot(current_data.timestamp, np.full(len(current_data.az), lower_threshold_az), color='olive', ls=('dotted'))
    ax[2,1].get_xaxis().set_major_locator(mdates.HourLocator(interval=6))
    ax[2,1].get_xaxis().set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[2,1].get_xaxis().label.set_fontsize(10)
    ax[2,0].set_ylim([min_az, max_az])
    ax[2,0].set_ylabel("acceleration (Z axis)", fontsize=30)
    
    segments_ax = np.empty(len(current_data.ax))
    segments_ax[:] = np.nan
    segments_ay = np.empty(len(current_data.ay))
    segments_ay[:] = np.nan
    segments_az = np.empty(len(current_data.az))
    segments_az[:] = np.nan
    
    for segment in current_segments:
        segments_ax[int(segment.start):int(segment.end)] = current_data.ax[int(segment.start):int(segment.end)]
        segments_ay[int(segment.start):int(segment.end)] = current_data.ay[int(segment.start):int(segment.end)]
        segments_az[int(segment.start):int(segment.end)] = current_data.az[int(segment.start):int(segment.end)]
        
    ax[0,1].plot(current_data.timestamp, segments_ax, color='lightseagreen')
    ax[0,1].set_ylim([min_ax, max_ax])
    ax[1,1].plot(current_data.timestamp, segments_ay, color='coral')
    ax[1,1].set_ylim([min_ay, max_ay])
    ax[2,1].plot(current_data.timestamp, segments_az, color='olive')
    ax[2,1].set_ylim([min_az, max_az])

    ax[2,0].set_xlabel("datetime",fontsize=30)
    ax[2,1].set_xlabel("datetime",fontsize=30)
    plt.show()
    ##############################################
    ##############################################
    '''
i = 0
for segment in all_segments:
    segment.id = i
    i = i+1
   
print("Total number of segments: "+str(len(all_segments)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 2: Find some metrics for the detected segments.
import numpy as np

length_segments_1axis = []
length_segments_2axis = []
length_segments_3axis = []
length_segments = []
for segment in all_segments:
    if len(segment.axis) == 1:
        length_segments_1axis.append(len(segment.ax))
    if len(segment.axis) == 2:
        length_segments_2axis.append(len(segment.ax))
    if len(segment.axis) == 3:
        length_segments_3axis.append(len(segment.ax))
    length_segments.append(len(segment.ax))
        
print("Mean segment length: "+str(np.mean(length_segments)))

print("Number of 1-axis segments: "+str(len(length_segments_1axis)))    
print("Max 1-axis segment length: "+str(max(length_segments_1axis)))
print("Min 1-axis segment length: "+str(min(length_segments_1axis)))
print("Mean 1-axis segment length: "+str(np.mean(length_segments_1axis)))
print("")

print("Number of 2-axis segments: "+str(len(length_segments_2axis)))
print("Max 2-axis segment length: "+str(max(length_segments_2axis)))
print("Min 2-axis segment length: "+str(min(length_segments_2axis)))
print("Mean 2-axis segment length: "+str(np.mean(length_segments_2axis)))
print("")

print("Number of 3-axis segments: "+str(len(length_segments_3axis)))
print("Max 3-axis segment length: "+str(max(length_segments_3axis)))
print("Min 3-axis segment length: "+str(min(length_segments_3axis)))
print("Mean 3-axis segment length: "+str(np.mean(length_segments_3axis)))

#%% Cell 2.1: Plot segment length histogram
from matplotlib import pyplot as plt
import numpy as np

segment_lengths = []   
segment_lengths1 = []                
             
for segment in all_segments:
    if len(segment.ax) < 500:
        segment_lengths1.append(len(segment.ax))
        
for segment in all_segments:
    segment_lengths.append(len(segment.ax))

### Plot segment length histogram
hist, bin_edges = np.histogram(segment_lengths)

fig, ax = plt.subplots(2,1,figsize = (16,12))
ax[0].title.set_text("Segment length histogram (log scale).")
ax[0].hist(segment_lengths, bins=200, log=True, color = 'lightseagreen')
ax[0].axvline(x = 60, color = 'orangered')
ax[0].set_ylabel('Number of segments')
ax[0].set_xlabel('Segment length (number of samples)')
ax[1].title.set_text("Segment length histogram for segments smaller than 500 samples (log scale).")
ax[1].hist(segment_lengths1, bins=200, log=True, color = 'lightseagreen')
ax[1].axvline(x = 60, color = 'orangered')
ax[1].set_ylabel('Number of segments')
ax[1].set_xlabel('Segment length (number of samples)')
  # arguments are passed to np.histogram)

#%% Plots segmentation1
sigmas = [2, 4, 6, 8, 10]
num_segments = [337, 226, 663, 590, 481]
mean_len = [5760, 4156, 368, 208, 162]
comp_time = [7144, 743, 195, 196, 192]

fig, ax = plt.subplots(1,1,figsize = (8,6))
fig.suptitle("Number of segments as a function of sigma", fontsize = 25)
ax.plot(sigmas, num_segments, color = 'dodgerblue', marker = 'o')
ax.set_ylabel('Number of segments')
ax.set_xlabel('sigma')

fig, ax = plt.subplots(1,1,figsize = (8,6))
fig.suptitle("Mean segment length as a function of sigma", fontsize = 25)
ax.plot(sigmas, mean_len, color = 'chocolate', marker = 'o')
ax.set_ylabel('Mean segment length (number of samples)')
ax.set_xlabel('sigma')

fig, ax = plt.subplots(1,1,figsize = (8,6))
fig.suptitle("Computing time as a function of sigma", fontsize = 25)
ax.plot(sigmas, comp_time, color = 'olivedrab', marker = 'o')
ax.set_ylabel('Computing time (s)')
ax.set_xlabel('sigma')

#%% Plots segmentation2
sigmas = [10, 20, 50, 100, 300, 500]
num_segments = [1664, 1000, 663, 503, 291, 203]
mean_len = [78, 174, 368, 630, 1764, 3270]
comp_time = [199, 210, 177, 175, 186, 180]

fig, ax = plt.subplots(1,1,figsize = (8,6))
fig.suptitle("Number of segments as a function of w", fontsize = 25)
ax.plot(sigmas, num_segments, color = 'dodgerblue', marker = 'o')
ax.set_ylabel('Number of segments')
ax.set_xlabel('w')

fig, ax = plt.subplots(1,1,figsize = (8,6))
fig.suptitle("Mean segment length as a function of w", fontsize = 25)
ax.plot(sigmas, mean_len, color = 'chocolate', marker = 'o')
ax.set_ylabel('Mean segment length (number of samples)')
ax.set_xlabel('w')

fig, ax = plt.subplots(1,1,figsize = (8,6))
fig.suptitle("Computing time as a function of w", fontsize = 25)
ax.plot(sigmas, comp_time, color = 'olivedrab', marker = 'o')
ax.set_ylim([150, 250])
ax.set_ylabel('Computing time (s)')
ax.set_xlabel('w')