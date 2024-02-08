''' 23.01.2024 - LAB 1 - SCRIPT FOR Å GENERERE FFT AV SIGNAL SOM LESES FRA ADC '''

''' NOTEE: THE DC COMPONENT FROM THE ACOUSTIC SENSORS ARE 1.66V'''

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

filename = "Scope_ADC_1_2_rev1.csv"

with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        DA_data.append(values)

time1 = [p[0] for p in DA_data]
ch1 = [p[1] for p in DA_data]


def raspi_import(path, channels=5):

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data #The data-array generates a (31250, 5) array, meaning 5 channels


# Import data from bin file
if __name__ == "__main__":
    sample_period, data = raspi_import('C:/Users/bruker/OneDrive - NTNU/6. semester/TTT4280 Sensorer og instrumentering/Lab/Sensorer-og-instrumentering---Lab/Lab 1/Data/sampledData_200045.bin') #sampledData_101128 #sys.argv[1] or 
    dt = sample_period
    data = (data*3.308)/(2**12)  #Formel fra labhefte, skrive noe lurt om denne i rapporten. data*Vref/(4096)

    w = np.ones(len(data[:, 0:1]))

    for i in range(5):
        data[i, :] *= w[i]

    dc_comp = 1.66


def plot_ADC_channels(sample_period, data_inn):

    #Tidsakse
    t = np.arange(0, dt*len(data_inn), dt)

    #plt.plot(time1, ch1, linestyle = '--', label = 'lala', color = 'red')

    adc5_offset = [i+0.3 for i in data_inn[:, 4:5]]

    colours = ['blue', 'red', 'purple', 'forestgreen', 'coral']

    for i in range(1):
        plt.plot(t[1:]*10**3, data_inn[1:, i:(i+1)], label=f'$x_{i+1}[n]$', color = colours[i])#adc5_offset[1:200]

        plt.xlabel("Tid [ms]", fontsize=15)
        plt.ylabel("Amplitude [V]", fontsize=15)
        plt.title(f'Samplet signal av ADC {i+1}', fontsize = 17)
        plt.legend(loc="upper right", fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13) 
        plt.grid()
        plt.show()   


def func_FFT(data_inn): #data_inn er sanns 1 kolonne

    n = 2**int(np.ceil(np.log2(len(data_inn)))) - len(data_inn)
    data_inn = np.vstack((data_inn, np.zeros((n, len(data_inn)))))   

    #Beregner FFT for hver ADC kanal    
    data_FFT = np.fft.fft(data_inn, len(data_inn)) #n=2**int(np.ceil(np.log2(len(data))))

    #Frekvensakse
    freq = np.fft.fftfreq(n=len(data_FFT), d=sample_period)
    #freq = np.fft.fftshift(freq)+(fs/2)

    return freq, data_FFT


def plot_FFT(data_inn):
    #Hente ut frekvens- og FFT-data fra func_FFT funksjonen
    freq, data_FFT = func_FFT(data_inn[:, 0:1])
    
    plt.xlabel("Frekvens [Hz]", fontsize=15)
    plt.ylabel("Amplitude [V]", fontsize=15)
    plt.title("FFT av ADC signaler", fontsize=17)

    plt.plot(freq, (np.abs(data_FFT)/len(data_inn)))

    plt.legend(('$X_1(k)$'), loc="upper right", fontsize=15)
    plt.grid()
    plt.show()


def plot_periodogram(data_inn):
    #Hente ut frekvens- og FFT-data fra func_FFT funksjonen
    freq, data_FFT = func_FFT(data_inn[:, 0])

    plt.xlabel("Frekvens [Hz]", fontsize=15)
    plt.ylabel("Relativ effekt [dB]", fontsize=15)
    plt.title("Periodogram av $x_{1}[n]$ med zero-padding", fontsize=17)

    plt.plot(freq, 20*np.log10((np.abs(data_FFT))/np.max(np.abs(data_FFT))))

    plt.legend([f'$X_{1}(f)$'], loc="upper right", fontsize=15)
    plt.grid()
    #plt.show()

    return freq, data_FFT


def add_window():

    #Bartlett, blackman, hamming, hanningm, kaiser
    hanning = np.bartlett(len(data))

    for i in range(len(data)):

        data[i, :] *= hanning[i]

    #plot_FFT(data_with_hanning)
    freq, data_FFT_window = plot_periodogram(data)
    #plot_ADC_channels(sample_period, data)

    return freq, data_FFT_window


#############################################################################
# freq, data_FFT = plot_periodogram(data)
# freq, data_FFT_window = add_window()

# plt.xlabel("Frekvens [Hz]", fontsize=15)
# plt.ylabel("Relativ effekt [dB]", fontsize=15)
# plt.title("Periodogram av $x_{1}[n]$ med zero-padding", fontsize=17)

# plt.plot(freq, 20*np.log10((np.abs(data_FFT))/np.max(np.abs(data_FFT))))
# plt.plot(freq, 20*np.log10((np.abs(data_FFT_window))/np.max(np.abs(data_FFT_window))))

# plt.legend([f'$X_{1}(f)$'], loc="upper right", fontsize=15)
# plt.grid()
# plt.show()
#############################################################################


#plot_ADC_channels(sample_period, data[0:200])
#func_FFT(data)
#plot_FFT(data)
plot_periodogram(data)
#add_window()


