#%% Cell 1: Load data, filter signals and find segments.
import time
import copy
import warnings
import data as dt
import numpy as np
import data_manager
import segment_manager
import segment as sgmnt
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

start_time = time.time()

### Initialize data_managerager class.

### Initialize with sigma, w and mode (rest or mean).
sigma = 6
w = 50
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)
data_manager = data_manager.data_manager()
### Initialize a list to store the events from all the datasets.
all_data = []
all_segments = []

### Define the names of the datasets that we will use
filenames = ['8200487_PHAAET_rec04052019_PRincon_S1',
             '8200718_PHAAET_rec08032019_PRincon']

### Detect events for a given datasets
for filename in filenames:
    
    #path ='D:\\AdolfoAB\\cobas_infinity_3.02\\Rabijunco\\'+filename+'\\'
    path = 'C:\\Users\\adolf\\Documents\\Adolfo\\TFG\\Data\\Accelerometria\\Rabijunco\\'+filename+'\\'
    
    # Load data and filter acceleration signals with a butterworth filter
    initial_data = data_manager.load_data(filename, path)
    current_data = copy.deepcopy(initial_data)
    current_data.filter_accelerations(4, 0.4)
    all_data.append(current_data)
    print("Data loaded: "+filename)
    
    '''
    ###############################
    ### Plot raw vs filtered signal
    fig, ax = plt.subplots(3,2,figsize = (8,6))
    fig.suptitle("Raw vs filtered acceleration signals")
    ax[0,0].plot(initial_data.ax[10000:10200], 'b-')
    ax[0,0].set_ylim([-3, 3])
    ax[0,1].plot(current_data.ax[10000:10200], 'b-')
    ax[0,1].set_ylim([-3, 3])
    ax[0,0].set_ylabel("ax")
    ax[1,0].plot(initial_data.ay[10000:10200], 'g-')
    ax[1,0].set_ylim([-3, 3])
    ax[1,1].plot(current_data.ay[10000:10200], 'g-')
    ax[1,1].set_ylim([-3, 3])
    ax[1,0].set_ylabel("ay")
    ax[2,0].plot(initial_data.az[10000:10200], 'r-')
    ax[2,0].set_ylim([-3, 3])
    ax[2,1].plot(current_data.az[10000:10200], 'r-')
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
        segment.data = current_data
        segment.setup_acceleration()
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
    
    ### Remove segments shorter than threshold.
    min_segment_size = 25
    current_segments = segment_manager.remove_short_segments(current_segments, min_segment_size)
    print("Number of segments after removing short evernts: "+str(len(current_segments)))
    
    ### Add segment id to each segment.
    for segment in current_segments:
        segment.id = current_segments.index(segment)
    
    ### Export segments from filename to CSV
    export_path = "C:\\Users\\adolf\\TFG\\Output_16052021\\"
    data_manager.export_segments(current_segments, sigma, w, filename, export_path)
    print("Segments successfully exported to .csv.")
    print("")
    
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
    
    fig, ax = plt.subplots(3,2,figsize = (16,12))
    fig.suptitle("Original signal vs segmented signal")
    ax[0,0].plot(current_data.ax, 'b-')
    ax[0,0].plot(np.full(len(current_data.ax), upper_threshold_ax), 'b-', ls=('dotted'))
    ax[0,0].plot(np.full(len(current_data.ax), lower_threshold_ax), 'b-', ls=('dotted'))
    ax[0,1].plot(np.full(len(current_data.ax), upper_threshold_ax), 'b-', ls=('dotted'))
    ax[0,1].plot(np.full(len(current_data.ax), lower_threshold_ax), 'b-', ls=('dotted'))
    ax[0,0].set_ylim([min_ax, max_ax])
    ax[0,0].set_ylabel("ax")
    ax[1,0].plot(current_data.ay, 'g-')
    ax[1,0].set_ylim([min_ay, max_ay])
    ax[1,0].plot(np.full(len(current_data.ay), upper_threshold_ay), 'g-', ls=('dotted'))
    ax[1,0].plot(np.full(len(current_data.ay), lower_threshold_ay), 'g-', ls=('dotted'))
    ax[1,1].plot(np.full(len(current_data.ay), upper_threshold_ay), 'g-', ls=('dotted'))
    ax[1,1].plot(np.full(len(current_data.ay), lower_threshold_ay), 'g-', ls=('dotted'))
    ax[1,0].set_ylabel("ay")
    ax[2,0].plot(current_data.az, 'r-')
    ax[2,0].plot(np.full(len(current_data.az), upper_threshold_az), 'r-', ls=('dotted'))
    ax[2,0].plot(np.full(len(current_data.az), lower_threshold_az), 'r-', ls=('dotted'))
    ax[2,1].plot(np.full(len(current_data.az), upper_threshold_az), 'r-', ls=('dotted'))
    ax[2,1].plot(np.full(len(current_data.az), lower_threshold_az), 'r-', ls=('dotted'))
    ax[2,0].set_ylim([min_az, max_az])
    ax[2,0].set_ylabel("az")
    
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
        
    ax[0,1].plot(segments_ax, 'b-')
    ax[0,1].set_ylim([min_ax, max_ax])
    ax[1,1].plot(segments_ay, 'g-')
    ax[1,1].set_ylim([min_ay, max_ay])
    ax[2,1].plot(segments_az, 'r-')
    ax[2,1].set_ylim([min_az, max_az])

    ax[2,0].set_xlabel("Number of samples")
    ax[2,1].set_xlabel("Number of samples")
    plt.show()
    ##############################################
    ##############################################
    '''
i = 0
for segment in all_segments:
    segment.id = i
    i = i+1

### Export all segments to CSV
#export_path = "D:\\AdolfoAB\\cobas_infinity_3.02\\Output_12052021\\"
export_path = "C:\\Users\\adolf\\TFG\\Output_16052021\\"

data_manager.export_all_segments(all_segments, sigma, w, export_path)
print("All segments successfully exported to .csv.")
print("")
   
print("Total number of segments: "+str(len(all_segments)))
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 2: Find some useful metrics for the detected events.
import numpy as np

length_segments_1axis = []
length_segments_2axis = []
length_segments_3axis = []
for segment in all_segments:
    if len(segment.axis) == 1:
        length_segments_1axis.append(len(segment.ax))
    if len(segment.axis) == 2:
        length_segments_2axis.append(len(segment.ax))
    if len(segment.axis) == 3:
        length_segments_3axis.append(len(segment.ax))

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
for segment in all_segments:
    if len(segment.ax) < 2000:
        segment_lengths.append(len(segment.ax))

### Plot segment length histogram
hist, bin_edges = np.histogram(segment_lengths)

fig, ax = plt.subplots(1,1,figsize = (8,6))
ax.title.set_text("Event length histogram.")
ax.hist(segment_lengths, bins=200, log=True)  # arguments are passed to np.histogram)

#%% Cell 3.1: Set up some values for plotting.
import numpy as np

### Set upper and lower thresholds for plotting
for segmentdata in all_data:
    std_ax = np.std(segmentdata.ax)
    std_ay = np.std(segmentdata.ay)
    std_az = np.std(segmentdata.az)
    
    if segment_manager.mode == "mean":
        mean_ax = np.mean(segmentdata.ax)
        mean_ay = np.mean(segmentdata.ay)
        mean_az = np.mean(segmentdata.az)
        
        upper_threshold_ax = mean_ax + segment_manager.sigma*std_ax
        lower_threshold_ax = mean_ax - segment_manager.sigma*std_ax
        upper_threshold_ay = mean_ay + segment_manager.sigma*std_ay
        lower_threshold_ay = mean_ay - segment_manager.sigma*std_ay
        upper_threshold_az = mean_az + segment_manager.sigma*std_az
        lower_threshold_az = mean_az - segment_manager.sigma*std_az
        
    if segment_manager.mode == "rest":
        upper_threshold_ax = 0 + segment_manager.sigma*std_ax
        lower_threshold_ax = 0 - segment_manager.sigma*std_ax
        upper_threshold_ay = 0 + segment_manager.sigma*std_ay
        lower_threshold_ay = 0 - segment_manager.sigma*std_ay
        upper_threshold_az = 0 + segment_manager.sigma*std_az
        lower_threshold_az = 0 - segment_manager.sigma*std_az
        
    for segment in all_segments:
        if segment.filename == segmentdata.filename:
            segment.setup_thresholds(upper_threshold_ax, lower_threshold_ax, upper_threshold_ay, lower_threshold_ay, upper_threshold_az, lower_threshold_az)

### Set min and max values of acceleration for axis scaling in plotting
for data in all_data:
    min_ax, max_ax = min(data.ax)-1, max(data.ax)+1
    min_ay, max_ay = min(data.ay)-1, max(data.ay)+1
    min_az, max_az = min(data.az)-1, max(data.az)+1
    min_pressure, max_pressure = min(data.pressure)-10, max(data.pressure)+10
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.min_ax, segment.max_ax = min_ax, max_ax
            segment.min_ay, segment.max_ay = min_ay, max_ay
            segment.min_az, segment.max_az = min_az, max_az
            segment.min_pressure, segment.max_pressure = min_pressure, max_pressure

#%% Cell 3.2: Plot some of the segments.
from matplotlib import pyplot as plt
 
### Plot segments given a condition
j = 0
while j <= 5:
    for segment in all_segments:
        if len(segment.axis) == 3:
        #if segment.id == 245 or segment.id == 3:
            j += 1
            fig, ax = plt.subplots(4,1,figsize = (8,6))
            ax[0].title.set_text("Segment id: "+str(segment.id)+". Segment Axis: "+segment.axis)
            ax[0].plot(segment.ax, 'b-')
            ax[0].plot(np.full(len(segment.ax), segment.upper_threshold_ax), 'b-', ls=('dotted'))
            ax[0].plot(np.full(len(segment.ax), segment.lower_threshold_ax), 'b-', ls=('dotted'))
            ax[0].plot(np.full(len(segment.ax), 0), 'b-', ls=('dashed'), lw=0.5)
            ax[0].set_ylim([segment.min_ax, segment.max_ax])
            ax[0].set_ylabel("ax")
            ax[1].plot(segment.ay, 'g-')
            ax[1].plot(np.full(len(segment.ay), segment.upper_threshold_ay), 'g-', ls=('dotted'))
            ax[1].plot(np.full(len(segment.ay), segment.lower_threshold_ay), 'g-', ls=('dotted'))
            ax[1].plot(np.full(len(segment.ay), 0), 'g-', ls=('dashed'), lw=0.5)
            ax[1].set_ylim([segment.min_ay, segment.max_ay])
            ax[1].set_ylabel("ay")
            ax[2].plot(segment.az, 'r-')
            ax[2].plot(np.full(len(segment.az), segment.upper_threshold_az), 'r-', ls=('dotted'))
            ax[2].plot(np.full(len(segment.az), segment.lower_threshold_az), 'r-', ls=('dotted'))
            ax[2].plot(np.full(len(segment.ay), 0), 'g-', ls=('dashed'), lw=0.5)
            ax[2].set_ylim([segment.min_az, segment.max_az])
            ax[2].set_ylabel("az")
            ax[3].plot(segment.pressure, 'k-')
            ax[3].set_ylim([segment.min_pressure, segment.max_pressure])
            ax[3].set_xlabel("number of samples")
            ax[3].set_ylabel("pressure (mBar)")
            plt.show()


#%% Cell 4: Compute maxcorr between each segment.
import os
import numpy as np
import segment_manager
import time

'''
This cell allows to compute the max correlation and lag arrays. 
However, this is very computationally expensive and if the amount of segments is too big, it will take a long time.
In order to improve the performance of this process, we created the compute_corr.py file, which does the same thing but using
the multiprocessing package to take advantage of parallel processing. This cannot be done here because of some problematic interactions
between the multiprocessing package, IPython and Windows.

In order to run compute_corr.py, just open the compute_corr.py file 
and set up the proper paths and filenames where the acceleration data and segments are.
Then, open a cmd at the corresponding window and write "python "compute_corr.py"". 
The correlation and lag arrays will be exported as .npy files.
'''

start_time = time.time()
   
sigma = 6
w = 50
mode = "mean"
segment_manager = segment_manager.segment_manager(sigma, w, mode)

temp_segments = copy.deepcopy(all_segments)

i = 0
segments_ax, segments_ay, segments_az = [], [], []
for segment in temp_segments:
    segment.id = i
    segments_ax.append(np.array(segment.ax))
    segments_ay.append(np.array(segment.ay))
    segments_az.append(np.array(segment.az))
    i = i + 1

maxcorr_ax, lag_ax = segment_manager.compute_max_corr(segments_ax)
maxcorr_ay, lag_ay = segment_manager.compute_max_corr(segments_ay)
maxcorr_az, lag_az = segment_manager.compute_max_corr(segments_az)

### Save correlation and lag into numpy format
path = "C:\\Users\\adolf\\TFG\\Output_16052021\\" 
np.save(os.path.join(path, 'maxcorr_ax.npy'), maxcorr_ax)
np.save(os.path.join(path, 'maxcorr_ay.npy'), maxcorr_ay)
np.save(os.path.join(path, 'maxcorr_az.npy'), maxcorr_az)
np.save(os.path.join(path, 'lag_ax.npy'), lag_ax)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")


#%% Cell 6.0: Load events, correlation matrix and lag matrix
import numpy as np
import data_manager
from matplotlib import pyplot as plt

'''
This cell allows to load the previously created segments and the correlation and lag arrays without the need of running the pipeline again.
'''
data_manager = data_manager.data_manager()
### Load acceleration data
sigma = 6
w = 50
filenames = ['8200487_PHAAET_rec04052019_PRincon_S1',
             '8200718_PHAAET_rec08032019_PRincon']
all_data = []
for filename in filenames:
    #datapath ='D:\\AdolfoAB\\cobas_infinity_3.02\\Rabijunco\\'+filename+'\\'   
    datapath = 'C:\\Users\\adolf\\Documents\\Adolfo\\TFG\\Data\\Accelerometria\\Rabijunco\\'+filename+'\\'    
    # Load data and filter acceleration signals with a butterworth filter
    data = data_manager.load_data(filename, datapath)
    data.filter_accelerations(4, 0.4)
    all_data.append(data)
    print("Data loaded: "+filename)

### Load previously created segments
#path = "D:\\AdolfoAB\\cobas_infinity_3.02\\Output_15052021\\"
path = "C:\\Users\\adolf\\TFG\\Output_16052021\\"      
all_segments = data_manager.load_all_segments(path, sigma, w)
for data in all_data:
    for segment in all_segments:
        if segment.filename == data.filename:
            segment.data = data
            segment.setup_acceleration()

### Load correlation and lag arrays            
maxcorr_ax = np.load(path+"maxcorr_ax.npy")
maxcorr_ay = np.load(path+"maxcorr_ay.npy")
maxcorr_az = np.load(path+"maxcorr_az.npy")
lag_ax = np.load(path+"lag_ax.npy")

### Plot the max correlation arrays
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,figsize = (18,9))
im = ax[0].imshow(maxcorr_ax)
ax[0].set_xlabel("ax segment index")
ax[0].set_ylabel("ax segment index")
ax[1].imshow(maxcorr_ay)
ax[1].set_xlabel("ay segment index")
ax[1].set_ylabel("ay segment index")
ax[2].imshow(maxcorr_az)
ax[2].set_xlabel("az segment index")
ax[2].set_ylabel("az segment index")
fig.colorbar(im, ax=ax.ravel().tolist(), orientation = 'horizontal', aspect = 40)
fig.suptitle('Max correlation between each segment', y = 0.85)
plt.show()


#%% Cell 7: Group segments based on max correlation
import copy
import time
import segment_manager
start_time = time.time()

segment_manager = segment_manager.segment_manager(sigma, w)
threshold_ax = 0.37
threshold_ay = 0.1
threshold_az = 0.2

input_segments = copy.copy(all_segments)
similar_segments_groups = segment_manager.group_similar_segments(input_segments, maxcorr_ax, maxcorr_ay, maxcorr_az, threshold_ax, threshold_ay, threshold_az)

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 7.1: Remove the groups that have less than N elements.
import copy

segment_groups = []
temp_similar_groups = copy.copy(similar_segments_groups)
min_group_size = 50

for segmentgroup in temp_similar_groups:
    if len(segmentgroup) > min_group_size:
        segment_groups.append(segmentgroup)
        

#%% Cell 8: Align segments from the same group.
segments_to_align = copy.copy(segment_groups)
similar_segments_aligned = segment_manager.align_segments(segments_to_align, lag_ax)

for segmentdata in all_data:
    for segment_group in similar_segments_aligned:
        for segment in segment_group:
            if segment.filename == segmentdata.filename:
                segment.setup_acceleration()
        
#%% Cell 8.1: Compute metrics from aligned similar groups
import numpy as np
import eventgrouper as EventGrouper
grouper = EventGrouper.EventGrouper()

### Compute max group size
group_sizes = []
for segment_group in similar_segments_aligned:
    group_sizes.append(len(segment_group))
print("Maximum group size: "+str(max(group_sizes)))
print("Mean group size: "+str(np.mean(group_sizes)))

### Compute total number of segments after processing.
number_of_segments = 0
number_of_groups = 0
for segmentgroup in similar_segments_aligned:
    number_of_groups = number_of_groups + 1
    for segment in segmentgroup:
        number_of_segments = number_of_segments + 1
print("Total number of segments after processing: "+str(number_of_segments))
print("Total number of groups after processing: "+str(number_of_groups))

### Compute mean correlation coefficient (only takes account groups with size > 1)
temp_similar_groups = copy.deepcopy(similar_segments_aligned)
corr_coefs = []
for segment_group in temp_similar_groups:
    group_segment_sizes = []
    for segment in segment_group:
        group_segment_sizes.append(len(segment.ax))
    min_segment_size = min(group_segment_sizes)
    
    group_ax = []
    for segment in segment_group:
        segment.end = segment.start + min_segment_size
        
        for segmentdata in all_data:
            if segment.filename == segmentdata.filename:
                segment.data = segmentdata
                segment.setup_acceleration()
                
        group_ax.append(np.array(segment.ax))
    corr_coefs.append(np.mean(np.corrcoef(group_ax)))
        
print("Mean correlation coefficients: "+str(np.mean(corr_coefs)))

#%% Cell 8.2: Find average segment for every group.
import numpy as np
from matplotlib import pyplot as plt

'''
group_mean_ax, group_mean_ay, group_mean_az = [], [], []
for group in similar_events_aligned:
    group_ax, group_ay, group_az = [], [], []
    for event in group:
        group_ax.append(event.ax)
        group_ay.append(event.ay)
        group_az.append(event.az)
        
        max_len_ax = np.array([len(array) for array in group_ax]).max()
        max_len_ay = np.array([len(array) for array in group_ay]).max()
        max_len_az = np.array([len(array) for array in group_az]).max()
        
        group_ax = [np.pad(array, (0, max_len_ax - len(array)), mode='constant', constant_values=np.nan) for array in group_ax]
        group_ay = [np.pad(array, (0, max_len_ay - len(array)), mode='constant', constant_values=np.nan) for array in group_ay]
        group_az = [np.pad(array, (0, max_len_az - len(array)), mode='constant', constant_values=np.nan) for array in group_az]
'''
mean_group_ax, mean_group_ay, mean_group_az = [], [], []
for group in similar_segments_aligned:
    group_ax, group_ay, group_az = [], [], []
    for segment in group:
        group_ax.append(segment.ax)
        group_ay.append(segment.ay)
        group_az.append(segment.az)
        
        max_len_ax = np.array([len(array) for array in group_ax]).max()
        max_len_ay = np.array([len(array) for array in group_ay]).max()
        max_len_az = np.array([len(array) for array in group_az]).max()
        
    group_ax = np.array([np.pad(array, (0, max_len_ax - len(array)), mode='constant', constant_values=np.nan) for array in group_ax])
    group_ay = np.array([np.pad(array, (0, max_len_ay - len(array)), mode='constant', constant_values=np.nan) for array in group_ay])
    group_az = np.array([np.pad(array, (0, max_len_az - len(array)), mode='constant', constant_values=np.nan) for array in group_az])
    
    mean_group_ax.append(np.nanmean(group_ax, axis = 0))
    mean_group_ay.append(np.nanmean(group_ay, axis = 0))
    mean_group_az.append(np.nanmean(group_az, axis = 0))

for i in range(len(similar_segments_aligned)):
    max_ax = max(mean_group_ax[i])
    min_ax = min(mean_group_ax[i])
    max_ay = max(mean_group_ay[i])
    min_ay = min(mean_group_ay[i])
    max_az = max(mean_group_az[i])
    min_az = min(mean_group_az[i])
    
    
    fig, ax = plt.subplots(3,1,figsize = (8,6))
    ax[0].title.set_text("Group number "+str(i)+'. Group size = '+str(len(similar_segments_aligned[i])))
    ax[0].plot(mean_group_ax[i])
    ax[0].set_ylim([min_ax-1, max_ax+1])
    ax[1].plot(mean_group_ay[i])
    ax[1].set_ylim([min_ay-1, max_ay+1])
    ax[2].plot(mean_group_az[i])
    ax[2].set_ylim([min_az-1, max_az+1])

#%% Cell 9: Plot N segments from a group.
from matplotlib import pyplot as plt

def get_cmap(n, name='YlOrRd'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

groupnumber = 7
number_of_segments = 10000
cmap1 = get_cmap(len(similar_segments_aligned[groupnumber]))
cmap2 = get_cmap(len(similar_segments_aligned[groupnumber]), "Blues")

fig, ax = plt.subplots(2,1,figsize = (8,6))
i = 0
for segment in similar_segments_aligned[groupnumber]:
    if i < number_of_segments and len(segment.ax) < 1000000:
        ax[0].title.set_text("Group number "+str(groupnumber)+'. Group size = '+str(len(similar_segments_aligned[groupnumber])))
        ax[0].plot(segment.ax, c=cmap1(i), lw=0.8)
        ax[0].set_ylim([-9, 9])
        ax[1].plot(segment.pressure, c=cmap2(i), lw=0.8)
        ax[1].set_ylim([990,1200])
        #ax[0].plot(np.full(len(segment.ax), segment.upper_threshold_ax), 'b-', ls=('dotted'))
        #ax[0].plot(np.full(len(segment.ax), segment.lower_threshold_ax), 'b-', ls=('dotted'))
        #ax[1].plot(segment.ay, c=cmap(i))
        #ax[1].set_ylim([-9, 9])
        #ax[2].plot(segment.az, c=cmap(i))
        #ax[2].set_ylim([-9, 9])
        i = i+1
plt.show()

#%% Cell 10: Set group label to each segment and prepare the segments for reservoir computing model.
segments_processed = []
group_label = 0
for group in similar_segments_aligned:
    for segment in group:
        segment.group_label = group_label
        segments_processed.append(segment)
    group_label = group_label + 1
    

#%% Cell 12: Create training data for reservoir.
import numpy as np
import random
import copy

segments_train = []
segments_test = copy.deepcopy(segments_processed)

num_groups = len(similar_segments_aligned)
num_segments_train = 50
num_segments_test = len(segments_processed) - num_segments_train*num_groups  # Number of segments from each group that we will use to train the network.
labels_train = np.zeros(num_segments_train*num_groups)
labels_test = np.zeros(num_segments_test)
train_data_ax, train_data_ay, train_data_az, len_segments_train = [], [], [], []
test_data_ax, test_data_ay, test_data_az, len_segments_test = [], [], [], []           

group_min_lengths = []

for group in similar_segments_aligned:
    group_min_lengths.append(np.array([len(segment.ax) for segment in group]).min())

k = 0
for i in range(num_segments_train):
    for j in range(num_groups):
        for segment in segments_test:
            if segment.group_label == j:
                segments_train.append(segment)
                segments_test.remove(segment)
                len_segments_train.append(len(segment.ax))
                labels_train[k] = segment.group_label
                k = k + 1
                for ax in segment.ax:
                    train_data_ax.append(ax)
                for ay in segment.ay:
                    train_data_ay.append(ay)
                for az in segment.az:
                    train_data_az.append(az)   
                break

for segment in segments_train:
    segment.end = segment.start + group_min_lengths[segment.group_label]
    segment.ax = segment.ax[0:group_min_lengths[segment.group_label]]
    segment.ay = segment.ay[0:group_min_lengths[segment.group_label]]
    segment.az = segment.az[0:group_min_lengths[segment.group_label]]
            
for segment in segments_test:
    len_segments_test.append(len(segment.ax))
    labels_test[segments_test.index(segment)] = segment.group_label
    for ax in segment.ax:
        test_data_ax.append(ax)
    for ay in segment.ay:
        test_data_ay.append(ay)
    for az in segment.az:
        test_data_az.append(az)
        
train_data = np.array([train_data_ax, train_data_ay, train_data_az])
test_data = np.array([test_data_ax, test_data_ay, test_data_az])

        
#%% Cell 12.1: Create test data for reservoir.
import numpy as np
import random

num_groups = len(similar_segments_aligned)
num_segments_test = len(segments_processed) - len(segments_train) # All the segments that are not used in training, are used in the test.
labels_test = np.zeros((num_segments_test, num_groups))

for segment in segments_test:
    len_segments_test.append(len(segment.ax))
    labels_test[segments_test.index(segment), segment.group_label] = 0.9
    for ax in segment.ax:
        test_data_ax.append(ax)
    for ay in segment.ay:
        test_data_ay.append(ay)
    for az in segment.az:
        test_data_az.append(az)

        

#%% Cell 13: Trying reservoir with my own data.
import copy
import network2 as Network2

Network2 = Network2.Network()
num_nodes = 10

input_probability = 0.7
reservoir_probability = 0.7
classifier = "log"

Network2.T = sum(len_segments_train)  
Network2.n_min = 2540
Network2.K = 3
Network2.N = num_nodes

Network2.setup_network(train_data, num_nodes, input_probability, reservoir_probability, num_groups, num_segments_train*num_groups)
Network2.train_network(num_groups, classifier, num_segments_train*num_groups, len_segments_train, labels_train, num_nodes)

Network2.mean_test_matrix = np.zeros([Network2.N, num_segments_test])
Network2.test_network(test_data, num_segments_test, len_segments_test, num_nodes, num_groups, sum(len_segments_test))

if classifier == 'log':
	print(f'Performance using {classifier} : {Network2.regressor.score(Network2.mean_test_matrix.T,labels_test.T)}')
