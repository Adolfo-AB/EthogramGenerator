import csv
import pandas as pd
import data as Data

class DataManager():
    def __init__(self):
        pass
    
    ### Load acceleration data from .csv file
    def LoadData(self, filename, path):
        try:
            data_pd = pd.read_csv(path + filename + '.csv', header=0, sep=';')
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
        except:
            data_pd = pd.read_csv(path + filename + '.csv', header=0)
            accelerations = data_pd.loc[:, ['X', 'Y', 'Z']].values
            
        data = Data.Data(filename, accelerations[:,0], accelerations[:,1], accelerations[:,2])
        data.pressure = data_pd.loc[:, ['Pressure']].values 
        
        return data
        
    def LoadEvents(self, filename, path, sigma, w):
        csv_filename = "events_sigma"+str(sigma)+"_w"+str(w)+"_"+filename+".csv" 
        pathfile = path + csv_filename
        
        events = []
        with open(pathfile, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != None:
                i = 0
                for row in reader:
                    if i % 2 != 0:
                        events.append(row)
                    i = i + 1
            
        print("Total number of events: "+str(len(events))) 
        return events
    
    def LoadAllEvents(self, path, sigma, w):
        import event as Event
        csv_filename = "allevents_sigma"+str(sigma)+"_w"+str(w)+".csv" 
        pathfile = path + csv_filename
        
        events = []
        with open(pathfile, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != None:
                i = 0
                for row in reader:
                    if i % 2 != 0:
                        events.append(row)
                    i = i + 1
                    
        all_events = []
        for event in events:
            current_event = Event.Event(int(float(event[1])), int(float(event[2])), event[3], event[4])
            current_event.id = int(event[0])
            all_events.append(current_event)
        
        print("Total number of events loaded: "+str(len(events))) 
        return all_events
    
    def LoadCorrelationMatrix2(self, path, sigma, w, number_of_events):
        import csv
        import numpy as np
        from itertools import islice
        
        corr_ax_filename = 'corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        corr_ay_filename = 'corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        corr_az_filename = 'corr_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        lag_ax_filename = 'lag_ax_s'+str(sigma)+'_w'+str(w)+'.csv'
        
        corr_ax_pathfile = path + corr_ax_filename
        corr_ay_pathfile = path + corr_ay_filename
        corr_az_pathfile = path + corr_az_filename
        lag_ax_pathfile = path + lag_ax_filename
        
        corr_ax, corr_ay, corr_az, lag_ax = [], [], [], []
        
        corr_ax_csv = open(corr_ax_pathfile)
        corr_ax_reader = csv.reader(corr_ax_csv)
        corr_ax = list(corr_ax_reader)[0]
        length = len(corr_ax)
        corr_ax = np.array([corr_ax[x:x+number_of_events] for x in range(0, length, number_of_events)])
                    
        corr_ay_csv = open(corr_ay_pathfile)
        corr_ay_reader = csv.reader(corr_ay_csv)
        corr_ay = list(corr_ay_reader)[0]
        corr_ay = np.array([corr_ay[x:x+number_of_events] for x in range(0, length, number_of_events)])
                    
        corr_az_csv = open(corr_az_pathfile)
        corr_az_reader = csv.reader(corr_az_csv)
        corr_az = list(corr_az_reader)[0]
        corr_az = np.array([corr_az[x:x+number_of_events] for x in range(0, length, number_of_events)])
                    
        lag_ax_csv = open(lag_ax_pathfile)
        lag_ax_reader = csv.reader(lag_ax_csv)
        lag_ax = list(lag_ax_reader)[0]
        lag_ax = np.array([lag_ax[x:x+number_of_events] for x in range(0, length, number_of_events)])
                    
        return corr_ax, corr_ay, corr_az, lag_ax
    
    ### Method to export the events to .csv
    def ExportEventsToCSV(self, events, sigma, w, filename, path):
        fields = ['id', 'start', 'end', 'axis', 'filename']
        export_filename = path+"events_sigma"+str(sigma)+"_w"+str(w)+"_"+filename+".csv"
        
        with open(export_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for event in events:
                writer.writerow([event.id, event.start, event.end, event.axis, event.filename])
                
    ### Method to export all the events to .csv
    def ExportAllEventsToCSV(self, events, sigma, w, path):
        fields = ['id', 'start', 'end', 'axis', 'filename']
        export_filename = path+"allevents_sigma"+str(sigma)+"_w"+str(w)+".csv"
        
        with open(export_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for event in events:
                writer.writerow([event.id, event.start, event.end, event.axis, event.filename])