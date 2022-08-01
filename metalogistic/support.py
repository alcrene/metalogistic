import numpy as np


def tuple_to_dict(tuple):
	return {'cdf_ps': tuple[0], 'cdf_xs': tuple[1], 'lbound': tuple[2], 'ubound': tuple[3]}


def is_list_like(obj):
	return isinstance(obj, list) or (isinstance(obj,np.ndarray) and obj.ndim==1)


def is_numeric(obj):
	return isinstance(obj, (float, int, np.int32, np.int64)) or (isinstance(obj,np.ndarray) and obj.ndim==0)