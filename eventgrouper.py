import copy
import numpy as np
from scipy import signal

class EventGrouper():
    
    def __init__(self):
        pass
    
    def ComputeMaxCorrelationMatrix(self, segments):        
        total_maxcorr_ax, total_maxcorr_ay, total_maxcorr_az = [], [], []
        total_lag_ax, total_lag_ay, total_lag_az  = [], [], []
        
        for segment1 in segments:
            maxcorr_ax, maxcorr_ay, maxcorr_az = [], [], []
            maxcorr_lags_ax, maxcorr_lags_ay, maxcorr_lags_az  = [], [], []
            for segment2 in segments:
                a_ax = segment1.ax
                b_ax = segment2.ax
                
                a_ay = segment1.ay
                b_ay = segment2.ay
                
                a_az = segment1.az
                b_az = segment2.az
            
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
    
    def GroupSimilarEvents(self, input_segments, corr_ax, corr_ay, corr_az, threshold_ax, threshold_ay, threshold_az):
        ### Add a global index to each segment from 0 to len(segments)
        segments = copy.copy(input_segments)
        
        ### Take one segment e1, if the next one has a correlation higher than threshold, we put them into a separate list. 
        ### Repeat until there are no more segments with correlation higher than threshold for e1.
        similar_segments = []
        i = 0
        while i < len(segments):
            current_segment = copy.copy(segments[i])
            temp_similar_segments = [current_segment]
            j = i+1
            while j < len(segments):
                next_segment = copy.copy(segments[j])
                next_segment_index = next_segment.id
                
                c_ax = copy.copy(corr_ax[current_segment.id, next_segment.id])
                c_ay = copy.copy(corr_ay[current_segment.id, next_segment.id])
                c_az = copy.copy(corr_az[current_segment.id, next_segment.id])
                
                if float(c_ax) >= threshold_ax and float(c_ay) >= threshold_ay and float(c_az) >= threshold_az:
                    temp_similar_segments.append(next_segment)
                    segments.remove(segments[j])
                    j = i+1
                else:
                    j = j+1
                        
            else:
                similar_segments.append(temp_similar_segments)
                i = i+1
            
        return similar_segments
    
    def AlignSimilarEvents(self, similar_segments, full_corr_ax_lag):    
        similar_segments_aligned = []
        for i in range(0, len(similar_segments)):
            first_segment = copy.copy(similar_segments[i][0])
            first_segment_index = first_segment.id
            
            temp_similar_segments_aligned = []
            temp_similar_segments_aligned.append(first_segment)
            
            for j in range(1, len(similar_segments[i])):
                current_segment = copy.copy(similar_segments[i][j])
                current_segment_index = current_segment.id
                
                lag = copy.copy(full_corr_ax_lag[first_segment.id][current_segment.id])
                
                new_current_segment = copy.copy(current_segment)
                current_segment.start = int(float(new_current_segment.start)) - int(lag)
                current_segment.end = int(float(new_current_segment.end)) - int(lag)
                
                temp_similar_segments_aligned.append(current_segment)
            
            similar_segments_aligned.append(temp_similar_segments_aligned)
            
        print("Similar segments aligned.")
        return similar_segments_aligned