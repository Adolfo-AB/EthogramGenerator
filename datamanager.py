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
        
    
    ### Method to export the events to .csv
    def ExportEventsToCSV(self, events, sigma, w, filename, path):
        fields = ['id', 'start', 'end', 'axis', 'filename']
        export_filename = path+"events_sigma"+str(sigma)+"_w"+str(w)+"_"+filename+".csv"
        
        with open(export_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for event in events:
                writer.writerow([event.id, event.start, event.end, event.axis, event.filename])