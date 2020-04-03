import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_F0_contour(file):
	sound = parselmouth.Sound(file)
	pitch = sound.to_pitch()
	return pitch.selected_array['frequency']


def get_SQ25(contour):
	return np.quantile(contour, 0.25)

def get_SQ75(contour):
	return np.quantile(contour, 0.75)

def get_F0_median(contour):
	return np.median(contour)

def get_sdmedian(contour):
	return np.median(np.gradient(contour))

def get_IDR(contour):
	return get_SQ75(contour) - get_SQ25(contour)

def get_voiced_segments(contour):
	voiced_segments = []
	segment_values = []
	for elem in contour:
	    if elem == 0:
	        if len(segment_values)>0:
	            voiced_segments.append(segment_values)
	            segment_values = []
	    else:
	        segment_values.append(elem)
	return voiced_segments

def get_voiced_segment_range(voiced_segment):
	"""
	Mean of the range of voiced features
	"""
	range_list = []
	for elem in voiced_segment:
	    range_list.append(np.max(elem)-np.min(elem))
	return np.mean(range_list)

def get_max_voiced_curvature(contour):
	non_zero_elems = contour[contour!=0]
	return np.max(non_zero_elems)

