#%% Cell 1: Load data
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
    
finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 2: Trying rupture
from matplotlib import pyplot as plt
import ruptures as rpt
import numpy as np
import time

start_time = time.time()

# change point detection
model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=100, model=model).fit(data.ax)
my_bkps = algo.predict(20000)

# show results
rpt.show.display(data.ax, my_bkps, figsize=(10, 6))
plt.show()

finish_time = time.time()
total_time = finish_time - start_time
print("Computing time:",total_time, "seconds.")

#%% Cell 3: Plot
for i in range(1,len(my_bkps)):
    fig, ax = plt.subplots(1,1,figsize = (8,6))
    ax.plot(data.ax[my_bkps[i-1]:my_bkps[i]], 'b-')
    plt.show()
