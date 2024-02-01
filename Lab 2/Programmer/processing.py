import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt

def max_lag(x_1,x_2):
    ss.correlate(x_1,x_2)
    ss.correlation_lags(x_1,)