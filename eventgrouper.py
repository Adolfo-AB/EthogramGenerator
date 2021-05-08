import copy
import numpy as np
from scipy import signal

class EventGrouper():
    
    def __init__(self):
        pass
    
    def ComputeMaxCorrelationMatrix(self, events):        
        total_maxcorr_ax, total_maxcorr_ay, total_maxcorr_az = [], [], []
        total_lag_ax, total_lag_ay, total_lag_az  = [], [], []
        
        for event1 in events:
            maxcorr_ax, maxcorr_ay, maxcorr_az = [], [], []
            maxcorr_lags_ax, maxcorr_lags_ay, maxcorr_lags_az  = [], [], []
            for event2 in events:
                a_ax = event1.ax
                b_ax = event2.ax
                
                a_ay = event1.ay
                b_ay = event2.ay
                
                a_az = event1.az
                b_az = event2.az
            
                normalized_a_ax = np.float16((a_ax - np.mean(a_ax)) / (np.std(a_ax)))
                normalized_b_ax = np.float16((b_ax - np.mean(b_ax)) / (np.std(b_ax)))
                
                normalized_a_ay = np.float16((a_ay - np.mean(a_ay)) / (np.std(a_ay)))
                normalized_b_ay = np.float16((b_ay - np.mean(b_ay)) / (np.std(b_ay)))
                
                normalized_a_az = np.float16((a_az - np.mean(a_az)) / (np.std(a_az)))
                normalized_b_az = np.float16((b_az - np.mean(b_az)) / (np.std(b_az)))
                
                corr_ax = np.float32(np.correlate(normalized_a_ax, normalized_b_ax, 'full') / max(len(a_ax), len(b_ax)))
                maxcorr_ax.append(max(corr_ax))
                
                corr_ay = np.float32(np.correlate(normalized_a_ay, normalized_b_ay, 'full') / max(len(a_ay), len(b_ay)))
                maxcorr_ay.append(max(corr_ay))
                
                corr_az = np.float32(np.correlate(normalized_a_az, normalized_b_az, 'full') / max(len(a_az), len(b_az)))
                maxcorr_az.append(max(corr_az))
                
                lags_ax = signal.correlation_lags(normalized_a_ax.size, normalized_b_ax.size, mode="full")
                lag_ax = lags_ax[np.argmax(corr_ax)]
                maxcorr_lags_ax.append(lag_ax)
                
                lags_ay = signal.correlation_lags(normalized_a_ay.size, normalized_b_ay.size, mode="full")
                lag_ay = lags_ay[np.argmax(corr_ay)]
                maxcorr_lags_ay.append(lag_ay)
                
                lags_az = signal.correlation_lags(normalized_a_az.size, normalized_b_az.size, mode="full")
                lag_az = lags_az[np.argmax(corr_az)]
                maxcorr_lags_az.append(lag_az)
            
            total_maxcorr_ax.append(maxcorr_ax)
            total_maxcorr_ay.append(maxcorr_ay)
            total_maxcorr_az.append(maxcorr_az)
            
            total_lag_ax.append(maxcorr_lags_ax)
            total_lag_ay.append(maxcorr_lags_ay)
            total_lag_az.append(maxcorr_lags_az)
            
        return total_maxcorr_ax, total_maxcorr_ay, total_maxcorr_az, total_lag_ax, total_lag_ay, total_lag_az
    
    def GroupSimilarEvents(self, input_events, corr_ax, corr_ay, corr_az, threshold_ax, threshold_ay, threshold_az):
        ### Add a global index to each event from 0 to len(events)
        events = copy.copy(input_events)
        
        ### Take one event e1, if the next one has a correlation higher than threshold, we put them into a separate list. 
        ### Repeat until there are no more events with correlation higher than threshold for e1.
        similar_events = []
        i = 0
        while i < len(events):
            current_event = copy.copy(events[i])
            temp_similar_events = [current_event]
            j = i+1
            while j < len(events):
                next_event = copy.copy(events[j])
                next_event_index = next_event.id
                
                c_ax = copy.copy(corr_ax[current_event.id, next_event.id])
                c_ay = copy.copy(corr_ay[current_event.id, next_event.id])
                c_az = copy.copy(corr_az[current_event.id, next_event.id])
                
                if c_ax >= threshold_ax and c_ay >= threshold_ay and c_az >= threshold_az:
                    temp_similar_events.append(next_event)
                    events.remove(events[j])
                    j = i+1
                else:
                    j = j+1
                        
            else:
                similar_events.append(temp_similar_events)
                i = i+1
            
        return similar_events
    
    def AlignSimilarEvents(self, similar_events, full_corr_ax_lag):    
        similar_events_aligned = []
        for i in range(0, len(similar_events)):
            first_event = copy.copy(similar_events[i][0])
            first_event_index = first_event.id
            
            temp_similar_events_aligned = []
            temp_similar_events_aligned.append(first_event)
            
            for j in range(1, len(similar_events[i])):
                current_event = copy.copy(similar_events[i][j])
                current_event_index = current_event.id
                
                lag = copy.copy(full_corr_ax_lag[first_event.id][current_event.id])
                
                new_current_event = copy.copy(current_event)
                current_event.start = int(float(new_current_event.start)) - lag
                current_event.end = int(float(new_current_event.end)) - lag
                
                temp_similar_events_aligned.append(current_event)
            
            similar_events_aligned.append(temp_similar_events_aligned)
            
        print("Similar events aligned.")
        return similar_events_aligned