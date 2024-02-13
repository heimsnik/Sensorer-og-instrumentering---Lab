# Importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sys
import os
import csv
import scipy.signal as ss
import time

DA_data = []

filename = "Scope_CH1_CH2.csv"

with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        DA_data.append(values)

time1 = [p[0] for p in DA_data]
ch1 = [p[1] for p in DA_data]

#plt.plot(time,ch2, label='$s$(t)', color = 'orange')
plt.plot(time,ch1, label = '$x$(t)', color = 'red')


plt.grid()
plt.xlabel('Tid [s]', fontsize=15)
plt.ylabel('Spenning [V]', fontsize=15)
plt.legend(loc='upper right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=15) 