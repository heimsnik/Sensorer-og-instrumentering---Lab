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
    sample_period, data = raspi_import('C:/Users/bruker/OneDrive - NTNU/6. semester/TTT4280 Sensorer og instrumentering/Lab/Sensorer-og-instrumentering---Lab/Lab 1/Data/sampledData_163208.bin') #sampledData_101128 #sys.argv[1] or 
    dt = sample_period
    data = (data*3.308)/(2**12)  #Formel fra labhefte, skrive noe lurt om denne i rapporten. data*Vref/(4096)

    dc_comp = 1.66

def artificial_data():

    n = 31250

    t = np.arange(0, dt*n, dt)
    x = np.sin(2*np.pi*300*t) + np.sin(2*np.pi*2200*t) + np.sin(2*np.pi*6000*t)

    x_arr = np.array([x] * 5)

    x_transpose = np.transpose(x_arr)

    return x_transpose


def plot_data(data_plot):
    #Tidsakse for samplede signaler
    t = np.arange(0, dt*len(data_plot[:, 0]), dt)

    colours = ['blue', 'red', 'purple', 'forestgreen']

    for i in range(4):
        plt.plot(t[:]*10**3, data_plot[:, i:(i+1)], label=f'$x_{i+1}[n]$', color = colours[i])

        plt.xlabel("Tid [ms]", fontsize=15)
        plt.ylabel("Amplitude [V]", fontsize=15)
        plt.title(f'Samplet signal av ADC {i+1}', fontsize = 17)
        plt.legend(loc="upper right", fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13) 
        plt.grid()
        plt.show()  


def zero_pad(data_pad):
    #n = 2**int(np.ceil(np.log2(len(data_pad)))) - len(data_pad)
    n = 10000

    data_pad_copy = data_pad.copy()

    data_transposed = np.transpose(data_pad_copy)

    # Pad zeros to each row
    padded_rows = [np.concatenate((row, np.zeros(n))) for row in data_transposed]

    # Transpose the padded rows back to the original shape
    padded_data = np.transpose(padded_rows)

    return padded_data


def window(data_window):
    hanning = np.hanning(len(data_window[:, 0]))

    data_window_copy = data_window.copy()

    for i in range(len(data_window_copy[:, 0])):

        data_window_copy[i:i+1, :] *= hanning[i]

    return data_window_copy


#FFT for alle signaler
def FFT(data_FFT):

    data_FFT_copy = data_FFT.copy()

    FFT_ADC1 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 0], len(data_FFT_copy[1:, 0])))
    FFT_ADC2 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 1], len(data_FFT_copy[1:, 0])))
    FFT_ADC3 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 2], len(data_FFT_copy[1:, 0])))
    FFT_ADC4 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 3], len(data_FFT_copy[1:, 0])))
    FFT_ADC5 = np.fft.fftshift(np.fft.fft(data_FFT_copy[1:, 4], len(data_FFT_copy[1:, 0])))

    #Frekvensakse
    freq = np.fft.fftshift(np.fft.fftfreq(n=len(FFT_ADC1), d=sample_period))

    return FFT_ADC1, FFT_ADC2, FFT_ADC3, FFT_ADC4, FFT_ADC5, freq


def plot_FFT(data, data_window, data_padded):
    FFT_ADC1, FFT_ADC2, FFT_ADC3, FFT_ADC4, FFT_ADC5, freq = FFT(data)
    FFT_ADC1_window, FFT_ADC2_window, FFT_ADC3_window, FFT_ADC4_window, FFT_ADC5_window, freq_window = FFT(data_window)
    FFT_ADC1_padded, FFT_ADC2_padded, FFT_ADC3_padded, FFT_ADC4_padded, FFT_ADC5_padded, freq_padded = FFT(data_padded)

    plt.xlabel("Frekvens [Hz]", fontsize=15)
    plt.ylabel("Relativ effekt [dB]", fontsize=15)
    plt.title("Periodogram av $x_{1}[n]$", fontsize=17)
    plt.plot(freq, abs(FFT_ADC1))
    #plt.plot(freq_window, abs(FFT_ADC1_window), color = 'g')
    plt.plot(freq_padded, abs(FFT_ADC1_padded), color = 'r')
    #plt.xlim(-100,100)
    #plt.ylim(-140, 5)
    plt.legend([f'$X_{1}(f)$'], loc="upper right", fontsize=15)
    plt.grid()
    plt.show()


def plot_periodogram(data_original, data_window, data_padded):
    FFT_ADC1, FFT_ADC2, FFT_ADC3, FFT_ADC4, FFT_ADC5, freq = FFT(data_original)
    FFT_ADC1_window, FFT_ADC2_window, FFT_ADC3_window, FFT_ADC4_window, FFT_ADC5_window, freq_window = FFT(data_window)
    FFT_ADC1_padded, FFT_ADC2_padded, FFT_ADC3_padded, FFT_ADC4_padded, FFT_ADC5_padded, freq_padded = FFT(data_padded)

    plt.xlabel("Frekvens [Hz]", fontsize=15)
    plt.ylabel("Relativ effekt [dB]", fontsize=15)
    plt.title("Periodogram av $x_{1}[n]$", fontsize=17)
    plt.plot(freq, 20*np.log10((np.abs(FFT_ADC1)/max(abs(FFT_ADC1)))), label = 'X_1(f)')
    plt.plot(freq_window, 20*np.log10((np.abs(FFT_ADC1_window)/max(abs(FFT_ADC1_window)))), color = 'g')
    #plt.plot(freq_padded, 20*np.log10((np.abs(FFT_ADC1_padded)/max(abs(FFT_ADC1_padded)))), color = 'r', label = 'X_1(f) padded')
    plt.xlim(-1200,1200)
    plt.ylim(-120, 5)
    plt.legend(loc="upper right", fontsize=15)
    plt.grid()
    plt.show()

x = artificial_data()

#The original data zero-padded
data_padded = zero_pad(data)

#The original data windowed
data_window =  window(data)

#The windowed data zero-padded
data_window_padded = zero_pad(data_window)

#The padded data windowed
data_padded_window = window(data_padded)

plot_FFT(data, data_padded_window, data_padded_window)
#plot_periodogram(data, data_window, data_padded_window)

#plot_data(data)


 