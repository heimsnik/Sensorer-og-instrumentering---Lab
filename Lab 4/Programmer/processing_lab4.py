# Importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sys
import os
import csv
import scipy.signal as ss
import time

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
    sample_period, data = raspi_import('C:/Users/bruker/OneDrive - NTNU/6. semester/TTT4280 Sensorer og instrumentering/Lab/Sensorer-og-instrumentering---Lab/Lab 4/Data/speed1_data1.bin')
    dt = sample_period
    data = (data*3.308)/(2**12)  #Formel fra labhefte, skrive noe lurt om denne i rapporten. data*Vref/(4096)

    data = ss.detrend(data, axis=0)


def plot_data(data_plot):
    #Tidsakse for samplede signaler
    t = np.arange(0, dt*len(data_plot[:, 0]), dt)

    #time_analog, ch1 = analog_signal()
    #plt.plot(time_analog,ch1, label=f'$x_{1}(t)$', color = 'blue', linestyle='--')

    plt.plot(t[1:], data_plot[1:, 3], label=f'$IF_I$', color = 'blue')
    plt.plot(t[1:], data_plot[1:, 4], label=f'$IF_Q$', color = 'orange')

    plt.xlabel("Tid [s]", fontsize=17)
    plt.ylabel("Spenning [V]", fontsize=17)
    plt.title(f'Samplet signal av alle ADCer', fontsize = 19)
    plt.legend(loc="upper right", fontsize=15)
    #plt.xlim(0,0.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) 
    plt.grid()
    plt.show() 

def zero_pad(data_pad):
    n = 2**int(np.ceil(np.log2(len(data_pad)))) - len(data_pad)
    #n = 100000

    data_pad_copy = data_pad.copy()

    data_transposed = np.transpose(data_pad_copy)

    # Pad zeros to each row
    padded_rows = [np.concatenate((row, np.zeros(n))) for row in data_transposed]

    # Transpose the padded rows back to the original shape
    padded_data = np.transpose(padded_rows)

    return padded_data


def window(data_window):
    #Bartlett, blackman, hamming, hanningm, kaiser
    hanning = np.hanning(len(data_window[:, 0]))

    data_window_copy = data_window.copy()

    for i in range(len(data_window_copy[:, 0])):

        data_window_copy[i:i+1, :] *= hanning[i]

    return data_window_copy

#FFT for alle signaler
def FFT(data_FFT):

    data_FFT_copy = data_FFT.copy()

    # Perform FFT on ADC4 (real component)
    FFT_ADC4 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 3], len(data_FFT_copy[1:, 0])))

    # Perform FFT on ADC5 (imaginary component)
    FFT_ADC5 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 4], len(data_FFT_copy[1:, 0])))

    # Create frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(n=len(FFT_ADC4), d=sample_period))

    # Combine real and imaginary components into complex FFT
    FFT_complex = FFT_ADC4 + 1j * FFT_ADC5

    return FFT_complex, freq


def plot_FFT(data, data_window, data_padded):
    FFT_normal, freq = FFT(data)
    #FFT_window, freq_window = FFT(data_window)
    #FFT_padded, freq_padded = FFT(data_padded)

    plt.xlabel("Frekvens [Hz]", fontsize=15)
    plt.ylabel("y-akse", fontsize=15)
    plt.title("FFT av $x_{1}[n]$", fontsize=17)
    plt.plot(freq, abs(FFT_normal)/max(abs(FFT_normal)))
    #plt.plot(freq_window, abs(FFT_ADC4_window)//max(abs(FFT_ADC4_window)), color = 'g')
    #plt.plot(freq_padded, abs(FFT_ADC4_padded)/max(abs(FFT_ADC4_padded)), color = 'r')
    plt.xlim(-300,300)
    #plt.ylim(-140, 5)
    plt.legend([f'$IF_I$'], loc="upper right", fontsize=15)
    plt.grid()
    plt.show()


def plot_periodogram(data_original, data_window, data_padded):
    FFT_normal, freq = FFT(data_original)
    FFT_window, freq_window = FFT(data_window)
    FFT_padded, freq_padded = FFT(data_padded)

    plt.xlabel("Frekvens [Hz]", fontsize=17)
    plt.ylabel("Relativ effekt [dB]", fontsize=17)
    plt.title(f"Periodogram", fontsize=19)
    plt.plot(freq, 20*np.log10((np.abs(FFT_normal)/max(abs(FFT_normal)))), label = f'$IF_I$')
    #plt.plot(freq_window, 20*np.log10((np.abs(FFT_ADC4_window)/max(abs(FFT_ADC4_window)))), label = f'$X_{1}(f)$ med hanningvindu', color = 'g')
    #plt.plot(freq_padded, 20*np.log10((np.abs(FFT_ADC1_padded)/max(abs(FFT_ADC1_padded)))), color = 'r', label = f'$X_{1}(f)$ med zero-padding')
    #plt.xlim(-300,300)
    plt.ylim(-120, 5)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.legend(loc="upper right", fontsize=14)
    plt.grid()
    plt.show()    

data_window = window(data)
data_padded = zero_pad(data)

#plot_data(data)
#plot_FFT(data, data_window, data_padded)
plot_periodogram(data, data_window, data_padded)

