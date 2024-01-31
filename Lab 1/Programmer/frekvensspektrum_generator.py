
''' 23.01.2024 - LAB 1 - SCRIPT FOR Å GENERERE FFT AV SIGNAL SOM LESES FRA ADC  '''

''' TIL NESTE GANG: FIKSE ALLE TITLER, FIKSE PERIODOGRAM PLOT, LAGE WINDOW OG ZERO-PAD FUNKSJON '''

# Importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sys
import os


def raspi_import(path, channels=5):

    #script_dir = os.path.dirname(os.path.realpath(__file__))
    #file_path = os.path.join(script_dir, path)

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')*(3.3/4095)
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))[:]

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data #The data-array generates a (31250, 5) array, meaning 5 channels


def plot_ADC_channels(sample_period, data):
    #Tidsakse
    t = np.arange(0, dt*len(data), dt)

    #Returnerer antall kolonner i data-arrayet (5)
    channels = data.shape[1]

    #Lager en liste med offsets for å tydligere se signaler
    offset = np.arange(data.shape[1])

    plt.plot(t,data[:])

    plt.xlabel("t [s]")
    plt.ylabel("Amplitude [V]")
    plt.title("ADC signaler")
    plt.legend([f'ADC {i+1}' for i in range(data.shape[1])], loc="upper right")
    plt.grid()
    plt.show()   


def func_FFT(data):
    #Beregner FFT for hver ADC kanal    
    data_FFT =  np.fft.fft(data, axis=0)

    #Frekvensakse
    freq = np.fft.fftfreq(n=len(data), d=sample_period)

    return freq, data_FFT


def plot_FFT():
    #Hente ut frekvens- og FFT-data fra func_FFT funksjonen
    freq, data_FFT = func_FFT(data)
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("FFT of x(t) with FFTShift")

    plt.plot(freq, np.fft.fftshift(np.abs(data_FFT))/len(data))

    plt.legend(('ADC1','ADC2','ADC3','ADC4','ADC5'), loc="upper right")
    plt.grid()
    plt.show()

#plot_FFT()

def plot_periodogram():

    #Hente ut frekvens- og FFT-data fra func_FFT funksjonen
    freq, data_FFT = func_FFT(data)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relativ effekt, [dB]")
    plt.title("Periodogram av ADC-kanaler")

    plt.plot(freq, 20*np.log10(np.fft.fftshift(abs(data_FFT))))#/max(abs(data_FFT)))

    plt.legend(('ADC1','ADC2','ADC3','ADC4','ADC5'), loc="upper right")
    plt.grid()
    plt.show()

#plot_periodogram()

def add_window():

    freq, data_FFT = func_FFT(data)

    #Bartlett, blackman, hamming, hanningm, kaiser
    hanning = np.hanning(len(data))
    data_with_hanning = np.multiply(data, hanning)
    data_with_hanning_FFT =  np.fft.fft(data_with_hanning, axis=0)

    return 0

def zero_pad():

    return 0

# Import data from bin file
sample_period, data = raspi_import("sampledData_21-09-01.bin")


dt = sample_period
#f_s = 1/dt

plot_ADC_channels(sample_period, data)




