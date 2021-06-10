# EthogramGenerator
Code to generate an ethogram from tri-axial accelerometry signals.

## data class
Class that models our data including filenames, acceleration signals, pressure, timestamps.

## data_manager class
Class that performs all the data-related functions (load data, export data, etc)

## segment class
Class that models our segment objects, including indices, acceleration and pressure signals.

## segment_manager class
Class that contains all the segment-related functions including segmentation algorithm and grouping algorithm.

## compute_corr.py
Script that computes the calculation of the correlation matrices using parallel processing for better performance.

## group_segments.py
Script that performs the grouping algorithm (both the compute_corr and group_segment methods are duplicated inside segment_manager class, but these isolated scripts have been created so they can be run through cmd for better performance).

## network class
Class that models our Reservoir Computing Recurrent Neural Network (adapted from https://github.com/ArnauNaval/TFG_ReservoirComputing/blob/master/network.py)

## main.py
Script that contains the end-to-end pipeline.

## main_gps.py
Script to create the gps plots.
