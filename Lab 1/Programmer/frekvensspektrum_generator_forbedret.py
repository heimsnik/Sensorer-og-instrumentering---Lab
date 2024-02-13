# Importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sys
import os
import csv
import scipy.signal as ss
import time

start_time = time.time()

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

    data = ss.detrend(data, axis=0)

def artificial_data():

    n = 31250

    t = np.arange(0, dt*n, dt)
    x = np.sin(2*np.pi*300*t) + np.sin(2*np.pi*2200*t) + np.sin(2*np.pi*6000*t)

    x_arr = np.array([x] * 5)

    x_transpose = np.transpose(x_arr)

    return x_transpose

def analog_signal():

    header = []
    analog_data = []
    analog_data2 = []

    filepath1 = r'C:/Users/bruker/OneDrive - NTNU/6. semester/TTT4280 Sensorer og instrumentering/Lab/Sensorer-og-instrumentering---Lab/Lab 1/Data/Scope_and_ADC_measurements/Scope_CH1_CH2.csv'
    filepath2 = r'C:/Users/bruker/OneDrive - NTNU/6. semester/TTT4280 Sensorer og instrumentering/Lab/Sensorer-og-instrumentering---Lab/Lab 1/Data\Scope_and_ADC_measurements/Scope_CH3_CH4.csv'


    # Read data from the specified filepath
    with open(filepath1) as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for datapoint in csvreader:
            values = [float(value) for value in datapoint]
            analog_data.append(values)

    time1 = [p[0] for p in analog_data]
    ch1 = [p[1] for p in analog_data]
    ch2 = [p[2] for p in analog_data]

    # Read data from the specified filepath
    with open(filepath2) as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for datapoint in csvreader:
            values = [float(value) for value in datapoint]
            analog_data2.append(values)

    time2 = [p[0] for p in analog_data2]
    ch3 = [p[1] for p in analog_data2]
    ch4 = [p[2] for p in analog_data2]
    ch5 = [(ch4[i]+0.7) for i in range(len(ch4))]

    plt.plot(time1,ch1, label=f'$x_{1}(t)$', color = 'blue', linestyle='--')
    plt.plot(time1,ch2, label = f'$x_{2}(t)$', color = 'red', linestyle='--')
    plt.plot(time2,ch3, label=f'$x_{3}(t)$', color = 'purple', linestyle='--')
    plt.plot(time2,ch4, label =f'$x_{4}(t)$', color = 'forestgreen', linestyle='--')
    plt.plot(time2,ch5, label =f'$x_{5}(t)$', color = 'cyan', linestyle='--')

    plt.grid()
    plt.xlim(0, 0.1)
    plt.title(f'Analoge inngangssignaler til alle ADCer', fontsize = 19)
    plt.xlabel('Tid [s]', fontsize=17)
    plt.ylabel('Spenning [V]', fontsize=17)
    plt.legend(loc='upper right', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) 

    plt.show()

    return time1, ch1


def plot_data(data_plot):
    #Tidsakse for samplede signaler
    t = np.arange(0, dt*len(data_plot[:, 0]), dt)

    colours = ['blue', 'red', 'purple', 'forestgreen', 'cyan']

    print(np.transpose(data_plot))
    
    for i in range(len(data_plot[0, :])):
        if i == 4:
            data_plot[:, i:(i+1)] += 0.7
        elif i == 0:
            data_plot[:, i:(i+1)] += 0
        else:
            data_plot[:, i:(i+1)] += 0.06

    #time_analog, ch1 = analog_signal()
    #plt.plot(time_analog,ch1, label=f'$x_{1}(t)$', color = 'blue', linestyle='--')

    plt.plot(t, data_plot[:, 0], label=f'$x_{1}[n]$', color = colours[0])
    plt.plot(t, data_plot[:, 1], label=f'$x_{2}[n]$', color = colours[1])
    plt.plot(t, data_plot[:, 2], label=f'$x_{3}[n]$', color = colours[2])
    plt.plot(t, data_plot[:, 3], label=f'$x_{4}[n]$', color = colours[3])
    plt.plot(t, data_plot[:, 4], label=f'$x_{5}[n]$', color = colours[4])

    plt.xlabel("Tid [s]", fontsize=17)
    plt.ylabel("Spenning [V]", fontsize=17)
    plt.title(f'Samplet signal av alle ADCer', fontsize = 19)
    plt.legend(loc="upper right", fontsize=15)
    plt.xlim(0,0.1)
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
    plt.plot(freq, abs(FFT_ADC1)/max(abs(FFT_ADC1)))
    plt.plot(freq_window, abs(FFT_ADC1_window)//max(abs(FFT_ADC1_window)), color = 'g')
    #plt.plot(freq_padded, abs(FFT_ADC1_padded)/max(abs(FFT_ADC1_padded)), color = 'r')
    plt.xlim(-100,100)
    #plt.ylim(-140, 5)
    plt.legend([f'$X_{1}(f)$'], loc="upper right", fontsize=15)
    plt.grid()
    plt.show()


def plot_periodogram(data_original, data_window, data_padded):
    FFT_ADC1, FFT_ADC2, FFT_ADC3, FFT_ADC4, FFT_ADC5, freq = FFT(data_original)
    FFT_ADC1_window, FFT_ADC2_window, FFT_ADC3_window, FFT_ADC4_window, FFT_ADC5_window, freq_window = FFT(data_window)
    FFT_ADC1_padded, FFT_ADC2_padded, FFT_ADC3_padded, FFT_ADC4_padded, FFT_ADC5_padded, freq_padded = FFT(data_padded)

    plt.xlabel("Frekvens [Hz]", fontsize=17)
    plt.ylabel("Relativ effekt [dB]", fontsize=17)
    plt.title(f"Periodogram av $x_{1}[n]$", fontsize=19)
    plt.plot(freq, 20*np.log10((np.abs(FFT_ADC1)/max(abs(FFT_ADC1)))), label = f'$X_{1}(f)$')
    #plt.plot(freq_window, 20*np.log10((np.abs(FFT_ADC1_window)/max(abs(FFT_ADC1_window)))), label = f'$X_{1}(f)$ med hanningvindu', color = 'g')
    #plt.plot(freq_padded, 20*np.log10((np.abs(FFT_ADC1_padded)/max(abs(FFT_ADC1_padded)))), color = 'r', label = f'$X_{1}(f)$ med zero-padding')
    plt.xlim(-300,300)
    plt.ylim(-120, 5)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.legend(loc="upper right", fontsize=14)
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

#plot_FFT(data, data_window, data_padded)
plot_periodogram(data, data_window, data_window_padded)

#plot_data(data)



 