import csv
import copy
import numpy as np
import event as Event
import more_itertools as mit

class EventFinder():
    def __init__(self, sigma, w, mode = "rest"):
        
        self.sigma = sigma #Beyond sigma*stdev we consider that we have an event or a potential behavior.
        self.w = w #Size of the window that we will use to find overlapping events.
        self.mode = mode

        
    def FindEvents(self, filename, a, axis, mode):
        stdev = np.std(a)
    
        if self.mode == "mean":
            mean = np.mean(a)
            index_events = np.where((a >= mean + self.sigma*stdev) | (a <= mean - self.sigma*stdev))
        if self.mode == "rest":
            index_events = np.where((a >= self.sigma*stdev) | (a <= -self.sigma*stdev))
        
        index_events = np.array(index_events)[0]
        raw_events = np.array([tuple(group) for group in mit.consecutive_groups(sorted(index_events))])

        events = []
        for raw_event in raw_events:
            events.append(Event.Event(raw_event[0], raw_event[len(raw_event)-1], axis, filename))
            
        return events
    
    
    def FindOverlappingEvents(self, filename, events, signal_length):
        events.sort(key=lambda event: event.start)
        i = 0
        while i < len(events):
    
            ### Current event --> event + window length i.e (start-w/2, end+w/2)
            current_event = self.AddWindowLength(events[i], signal_length)
            events[i] = current_event
            
            ### Find previous and next events
            previous_event = self.FindPreviousEvent(current_event, events)
            next_event = self.FindNextEvent(current_event, events)
            
            previous_overlapping = self.AreEventsOverlapping(current_event, previous_event)
            next_overlapping = self.AreEventsOverlapping(current_event, next_event)
            
            ### While previous or next event overlap with current event:
            while next_overlapping == True or previous_overlapping == True:
                while previous_overlapping == True:
                    new_event = self.MergeEvents(filename, current_event, previous_event)
                    
                    ### Check if the event indexes have changed. If so, add window length
                    event_changed = self.HasEventChanged(current_event, new_event)
                    
                    events, current_event, next_event, previous_event, i = self.UpdateEvents_Previous(events, current_event, previous_event, new_event)
                    if event_changed == True:
                        events[events.index(current_event)] = self.AddWindowLength(current_event, signal_length)
                        current_event = events[events.index(current_event)]
                        
                    previous_overlapping = self.AreEventsOverlapping(current_event, previous_event)
                    next_overlapping = self.AreEventsOverlapping(current_event, next_event)
                    
                while next_overlapping == True:
                    new_event = self.MergeEvents(filename, current_event, next_event)
                    
                    ### Check if the event indexes have changed. If so, add window length
                    event_changed = self.HasEventChanged(current_event, new_event)
                    
                    events, current_event, next_event, previous_event, i = self.UpdateEvents_Next(events, current_event, next_event, new_event)
                    if event_changed == True:
                        events[events.index(current_event)] = self.AddWindowLength(current_event, signal_length)
                        current_event = events[events.index(current_event)]
                    
                    previous_overlapping = self.AreEventsOverlapping(current_event, previous_event)
                    next_overlapping = self.AreEventsOverlapping(current_event, next_event)         
            
            else:
                i = i + 1
                
        return events
   
    
    ### Apply a window of size w to a given event in a way that an event (start, end) becomes (start - w/2, end + w/2).
    def AddWindowLength(self, event, signal_length):
        if event.start - self.w/2 < 0:
            event.start = 0
        else:
            event.start = event.start - self.w/2
            
        if event.end + self.w/2 > signal_length:
            event.end = signal_length
        else:
            event.end = event.end + self.w/2
            
        return event  
    
    
    ### Method to merge two overlapping events from the same dataset 
    def MergeEvents(self, filename, current_event, overlapping_event): 
        start = min([current_event.start, overlapping_event.start])
        end = max([current_event.end, overlapping_event.end])
        axis = ''.join(sorted(set(current_event.axis + overlapping_event.axis)))
        new_event = Event.Event(start, end, axis, filename)
        
        return new_event
    
    
    ### Method to find the previous event given a current event and a list of events that contains it.
    def FindPreviousEvent(self, current_event, events):
        if events.index(current_event)-1 >= 0:
            previous_event = events[events.index(current_event)-1]
        else:
            previous_event = None
        return previous_event
    
    
    ### Method to find the next event given a current event and a list of events that contains it.
    def FindNextEvent(self, current_event, events):
        if events.index(current_event)+1 <= len(events)-1:
            next_event = events[events.index(current_event)+1]
        else:
            next_event = None
        return next_event
    
    
    ### Check if two events are overlapping.
    def AreEventsOverlapping(self, current_event, other_event):
        if other_event != None and current_event != None:
            starts = [current_event.start, other_event.start]
            ends = [current_event.end, other_event.end]
    
            is_overlapping = True        
            for start in starts:
                for end in ends:
                    if start > end:
                        is_overlapping = False
        else:
            is_overlapping = False
                
        return is_overlapping
    
    
    ### Check if the start or the end of an event have changed.
    def HasEventChanged(self, current_event, new_event):
        event_changed = False
        if (current_event.start != new_event.start) or (current_event.end != new_event.end):
            event_changed = True
        
        return event_changed
    
    
    ### Update events list, current_event and previous_event.
    def UpdateEvents_Previous(self, events, current_event, previous_event, new_event):
        temp_previous_event = copy.copy(previous_event)
        try:
            events[events.index(current_event)] = new_event
        except:
            print("Error while reassigning the new event to its new position.")
            print(new_event)
                
        try:
            events.remove(previous_event)
        except:
            print("Error while trying to remove previous event.")
            print(temp_previous_event)
                
        events.sort(key=lambda event: event.start)
        current_event = new_event
        previous_event = self.FindPreviousEvent(current_event, events)
        next_event = self.FindNextEvent(current_event, events)
        i = events.index(current_event)
        return events, current_event, next_event, previous_event, i
    
    
    ### Update event list, current_event and next_event.
    def UpdateEvents_Next(self, events, current_event, next_event, new_event):
        temp_next_event = copy.copy(next_event)
        try:
            events[events.index(current_event)] = new_event
        except:
            print("Error while reassigning the new event to its new position.")
            print(new_event)
            
        try:
            events.remove(next_event)
        except:
            print("Error while trying to remove next event.")
            print(temp_next_event)
            
        events.sort(key=lambda event: event.start)
        current_event = new_event
        previous_event = self.FindPreviousEvent(current_event, events)
        next_event = self.FindNextEvent(current_event, events)
        i = events.index(current_event)
        return events, current_event, next_event, previous_event, i
    
    
    ### Removes all the events of size smaller than threshold.
    def RemoveShortEvents(self, events, threshold):
        new_events = []
        for event in events:
            length = event.end - event.start
            if length > threshold:
                new_events.append(event)
                
        return new_events
    
    
    ### Test to check that every event has the right axis label assigned.
    def TestCheckTagCoherence(self, events, data):
        
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
        for event in events:
            error_found = 0
            
            if (any(point > plus_ax or point < minus_ax for point in event.ax)) and ("x" not in event.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in event.ax:
                        if (point > plus_ax or point < minus_ax):
                            print("Coherence error.")
                            break
                    
            if (any(point > plus_ay or point < minus_ay for point in event.ay)) and ("y" not in event.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in event.ay:
                        if (point > plus_ay or point < minus_ay):
                            print("Coherence error.")
                            break
                    
            if (any(point > plus_az or point < minus_az for point in event.az)) and ("z" not in event.axis):
                if error_found == 0:
                    number_of_errors = number_of_errors + 1
                    error_found = 1
                    for point in event.az:
                        if (point > plus_az or point < minus_az):
                            print("Coherence error.")
                            break
                    
        return number_of_errors
    
    
    ### Test to check that every event detected initially can be found inside one of the events found after overlapping.
    def TestCheckEveryInitialEventIsInsideAFinalEvent(self, initial_events, final_events):
        number_of_errors = 0
        for initial_event in initial_events:
            event_found = 0
            for final_event in final_events:
                if initial_event.start >= final_event.start and initial_event.end <= final_event.end:
                    event_found = 1
                    break
            else:
                if event_found == 0:
                    number_of_errors = number_of_errors + 1
                    
        return number_of_errors
    
        