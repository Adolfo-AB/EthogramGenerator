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
            
        print("Total number of events: "+str(len(events))) 
        return events
    
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