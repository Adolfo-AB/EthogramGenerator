import copy
import numpy as np
import segment as sgmnt
from scipy import signal

import more_itertools as mit

class segment_manager():
    def __init__(self, sigma, w, mode = "rest"):
        
        self.sigma = sigma # Beyond sigma*stdev we consider that we have an segment or a potential behavior.
        self.w = w # Size of the window that we will use to find overlapping segments.
        self.mode = mode
        
    '''

    The function create_raw_segments creates raw segments from an acceleration signal. We define these initial segments as parts of the signal where:
        
        a < mean(a) - sigma*stdev(a) or a > mean(a) + sigma*stdev(a)
        
    Where a is the acceleration signal and sigma is the parameter that defines how much the acceleration has to deviate from its mean to
    segment it by that point.
    
    These segments are of the following form:
        
        segment = [start, end, axis]
    
    Where start is the sample at which the segment starts, end is the sample at which the segment ends, 
    and axis is the axis where the segment was initially created,
    
    '''    
    def create_raw_segments(self, filename, a, axis, mode):
        stdev = np.std(a)
    
        if self.mode == "mean":
            mean = np.mean(a)
            index_segments = np.where((a >= mean + self.sigma*stdev) | (a <= mean - self.sigma*stdev))
        if self.mode == "rest":
            index_segments = np.where((a >= self.sigma*stdev) | (a <= -self.sigma*stdev))
        
        index_segments = np.array(index_segments)[0]
        raw_segments = np.array([tuple(group) for group in mit.consecutive_groups(sorted(index_segments))])

        segments = []
        for segment in raw_segments:
            segments.append(sgmnt.segment(segment[0], segment[len(segment)-1], axis, filename))
            
        return segments
    
    '''
    
    The function overlap_segments attempts to convert the raw segments into segments that correspond to behaviors in a certain temporal scale.
    
    Given a list of segments sorted by their starting sample and given a current segment inside that list:
        - This function checks if applying a window of size w there is an overlapping with the previous and/or next segments.
        - If that's the case, it adds the window to the current segment and merges it with the previous or next segment (the one that overlaps) and repeat the previous step.
        - If
        - If not, go to the next segment and repreat the first step.
            
            For example, given a segment s1 = [start1, end1, "x"] and its previous segment s2 = [start2, end2, "yz"] and a window of size w.
                
                The function checks if applying the window to s1 there is an overlapping with s2:
                    start1' = start1 - w/2, end1' = end1 + w/2
                    s1' = [start1', end1', "x"]
                    
                If there is an overlapping, the function mergers both segments and repeats the process. Therefore:
                    s3 = [min(start1', start2, max(end1', end2), "xyz")]
   '''             
                    
    def overlap_segments(self, filename, segments, signal_length):
        segments.sort(key=lambda segment: segment.start)
        i = 0
        while i < len(segments):
    
            current_segment = segments[i]
            
            ### Find previous and next segments
            previous_segment = self.find_prev_segment(current_segment, segments)
            next_segment = self.find_next_segment(current_segment, segments)
            
            ### Check if there is overlappings between current segment and previous and/or next segments
            previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
            next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
            
            ### If there is no overlappings, check if applying the window there are overlappings
            if previous_overlapping != True and next_overlapping != True:
                current_segment == self.apply_window(current_segment, signal_length)
            
                previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                
            ### While previous or next segment overlap with current segment:
            while next_overlapping == True or previous_overlapping == True:
                while previous_overlapping == True:
                    new_segment = self.merge_segments(filename, current_segment, previous_segment)
                    
                    ### Check if the segment indexes have changed. If so, add window length
                    #segment_changed = self.HasSegmentChanged(current_segment, new_segment)
                    
                    segments, current_segment, next_segment, previous_segment, i = self.update_segments(segments, current_segment, previous_segment, next_segment, new_segment, "previous")
                    previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                    next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                    
                    if previous_overlapping != True and next_overlapping != True:
                        current_segment = self.apply_window(current_segment, signal_length)
                        previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                        next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                        
                while next_overlapping == True:
                    new_segment = self.merge_segments(filename, current_segment, next_segment)
                    
                    ### Check if the segment start or end have changed. If so, check if applying the window there will be an overlap with previous or next segments. 
                    #segment_changed = self.HasSegmentChanged(current_segment, new_segment)
                    
                    segments, current_segment, next_segment, previous_segment, i = self.update_segments(segments, current_segment, previous_segment, next_segment, new_segment, "next")
                    previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                    next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                    
                    if previous_overlapping != True and next_overlapping != True:
                        current_segment = self.apply_window(current_segment, signal_length)
                        previous_overlapping = self.are_segments_overlapping(current_segment, previous_segment)
                        next_overlapping = self.are_segments_overlapping(current_segment, next_segment)
                        
                               
            
            else:
                i = i + 1
                
        return segments
   
    
    ### Apply a window of size w to a given segment in a way that an segment (start, end) becomes (start - w/2, end + w/2).
    def apply_window(self, segment, signal_length):
        if segment.start - self.w/2 < 0:
            segment.start = 0
        else:
            segment.start = segment.start - self.w/2
            
        if segment.end + self.w/2 > signal_length:
            segment.end = signal_length
        else:
            segment.end = segment.end + self.w/2
            
        return segment  
    
    
    ### Method to merge two overlapping segments from the same dataset 
    def merge_segments(self, filename, current_segment, overlapping_segment): 
        start = min([current_segment.start, overlapping_segment.start])
        end = max([current_segment.end, overlapping_segment.end])
        axis = ''.join(sorted(set(current_segment.axis + overlapping_segment.axis)))
        new_segment = sgmnt.segment(start, end, axis, filename)
        
        return new_segment
    
    
    ### Method to find the previous segment given a current segment and a list of segments that contains it.
    def find_prev_segment(self, current_segment, segments):
        if segments.index(current_segment)-1 >= 0:
            previous_segment = segments[segments.index(current_segment)-1]
        else:
            previous_segment = None
        return previous_segment
    
    
    ### Method to find the next segment given a current segment and a list of segments that contains it.
    def find_next_segment(self, current_segment, segments):
        if segments.index(current_segment)+1 <= len(segments)-1:
            next_segment = segments[segments.index(current_segment)+1]
        else:
            next_segment = None
        return next_segment
    
    
    ### Check if two segments are overlapping.
    def are_segments_overlapping(self, current_segment, other_segment):
        if other_segment != None and current_segment != None:
            starts = [current_segment.start, other_segment.start]
            ends = [current_segment.end, other_segment.end]
    
            is_overlapping = True        
            for start in starts:
                for end in ends:
                    if start > end:
                        is_overlapping = False
        else:
            is_overlapping = False
                
        return is_overlapping
    
    
    ### Check if the start or the end of an segment have changed.
    def has_segment_changed(self, current_segment, new_segment):
        segment_changed = False
        if (current_segment.start != new_segment.start) or (current_segment.end != new_segment.end):
            segment_changed = True
        
        return segment_changed
    
    
    ### Update segments list
    def update_segments(self, segments, current_segment, previous_segment, next_segment, new_segment, previous_or_next):
        try:
            segments[segments.index(current_segment)] = new_segment
        except:
            print("Error while reassigning the new segment to its new position.")
            print(new_segment)
        
        if previous_or_next == "previous":
            try:
                segments.remove(previous_segment)
            except:
                print("Error while trying to remove previous segment.")
                print(previous_segment)
                
        elif previous_or_next == "next":
            try:
                segments.remove(next_segment)
            except:
                print("Error while trying to remove next segment.")
                print(next_segment)
                
        segments.sort(key=lambda segment: segment.start)
        current_segment = new_segment
        previous_segment = self.find_prev_segment(current_segment, segments)
        next_segment = self.find_next_segment(current_segment, segments)
        i = segments.index(current_segment)
        return segments, current_segment, next_segment, previous_segment, i
    
    '''
    ### Update segment list
    def update_segments_next(self, segments, current_segment, next_segment, new_segment):
        temp_next_segment = copy.deepcopy(next_segment)
        try:
            segments[segments.index(current_segment)] = new_segment
        except:
            print("Error while reassigning the new segment to its new position.")
            print(new_segment)
            
        try:
            segments.remove(next_segment)
        except:
            print("Error while trying to remove next segment.")
            print(temp_next_segment)
            
        segments.sort(key=lambda segment: segment.start)
        current_segment = new_segment
        previous_segment = self.find_prev_segment(current_segment, segments)
        next_segment = self.find_next_segment(current_segment, segments)
        i = segments.index(current_segment)
        return segments, current_segment, next_segment, previous_segment, i
    '''
    
    
    ### Removes all the segments of size smaller than threshold.
    def remove_short_segments(self, segments, min_segment_size):
        new_segments = []
        for segment in segments:
            length = segment.end - segment.start
            if length > min_segment_size:
                new_segments.append(segment)
                
        return new_segments
    
    
    ### Test to check that every segment has the right axis label assigned.
    def test_tag_coherence(self, segments, data):
        
        if self.mode == "mean":
            mean_ax = np.mean(data.ax)
            stdev_ax = np.std(data.ax)
            plus_ax = mean_ax + self.sigma*stdev_ax
            minus_ax = mean_ax - self.sigma*stdev_ax
            
            mean_ay = np.mean(data.ay)
            stdev_ay = np.std(data.ay)
            plus_ay = mean_ay + self.sigma*stdev_ay
            minus_ay = mean_ay - self.sigma*stdev_ay
            
            mean_az = np.mean(data.az)
            stdev_az = np.std(data.az)
            plus_az = mean_az + self.sigma*stdev_az
            minus_az = mean_az - self.sigma*stdev_az
        
        if self.mode == "rest":
            stdev_ax = np.std(data.ax)
            plus_ax = 0 + self.sigma*stdev_ax
            minus_ax = 0 - self.sigma*stdev_ax
            
            stdev_ay = np.std(data.ay)
            plus_ay = 0 + self.sigma*stdev_ay
            minus_ay = 0 - self.sigma*stdev_ay
            
            stdev_az = np.std(data.az)
            plus_az = 0 + self.sigma*stdev_az
            minus_az = 0 - self.sigma*stdev_az
        
        number_of_errors = 0
        for segment in segments:
            error_found = 0
            
            if (any(point > plus_ax or point < minus_ax for point in segment.ax)) and ("x" not in segment.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in segment.ax:
                        if (point > plus_ax or point < minus_ax):
                            print("Coherence error.")
                            break
                    
            if (any(point > plus_ay or point < minus_ay for point in segment.ay)) and ("y" not in segment.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in segment.ay:
                        if (point > plus_ay or point < minus_ay):
                            print("Coherence error.")
                            break
                    
            if (any(point > plus_az or point < minus_az for point in segment.az)) and ("z" not in segment.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in segment.az:
                        if (point > plus_az or point < minus_az):
                            print("Coherence error.")
                            break
                    
        return number_of_errors
    
    
    ### Test to check that every segment detected initially can be found inside one of the segments found after overlapping.
    def test_no_segments_missing(self, initial_segments, final_segments):
        number_of_errors = 0
        for initial_segment in initial_segments:
            segment_found = 0
            for final_segment in final_segments:
                if initial_segment.start >= final_segment.start and initial_segment.end <= final_segment.end:
                    segment_found = 1
                    break
            else:
                if segment_found == 0:
                    number_of_errors = number_of_errors + 1
                    
        return number_of_errors
    
     
    def compute_max_correlation(self, segments):        
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