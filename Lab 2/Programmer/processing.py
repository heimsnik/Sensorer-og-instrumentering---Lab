''' 23.01.2024 - LAB 1 - SCRIPT FOR Å GENERERE FFT AV SIGNAL SOM LESES FRA ADC '''

''' NOTEE: THE DC COMPONENT FROM THE ACOUSTIC SENSORS ARE 1.66V'''

# Importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sys
import os
import scipy.signal as ss


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
    sample_period, data = raspi_import('C:/Users/cmhei/OneDrive/Dokumenter/Semester 6/TTK4280 Sensorer og instrumentering/Lab/Lab 2/Data/Calibration_180_deg/sampledData_111909.bin') #sampledData_101128 #sys.argv[1] or 
    dt = sample_period
    fs = 1/dt
    data = (data*3.308)/(2**12)  #Formel fra labhefte, skrive noe lurt om denne i rapporten. data*Vref/(4096)

    dc_comp = 1.66

    #Removing dc_comp for each element in data-array
    for i in range(len(data[:, 0:1])):

        data[i:i+1, :] -= dc_comp
    
#sampledData_104504.bin 180
#sampledData_104519.bin 150
#sampledData_104533.bin 120

'''             SYSTEMOPPSETT LAB           '''
def plot_ADC_channels(sample_period, data):


    #Tidsakse
    t = np.arange(0, dt*len(data), dt)

    #Returnerer antall kolonner i data-arrayet (5)
    channels = data.shape[1]
    #Lager en liste med offsets for å tydligere se signaler
    offset = np.arange(data.shape[1])

    plt.plot(t[1:], data[1:, 0:3])#-(offset/10))

    plt.xlabel("Tid [s]")
    plt.ylabel("Amplitude [V]")
    plt.title("ADC signaler")
    plt.legend([f'$ADC {i+1}$' for i in range(data.shape[1])], loc="upper right")
    plt.grid()
    plt.show()   


def func_FFT(data):

    #Beregner FFT for hver ADC kanal    
    data_FFT = np.fft.fft(data, n=2**int(np.ceil(np.log2(len(data)))), axis=0)

    #Frekvensakse
    freq = np.fft.fftfreq(n=len(data_FFT), d=sample_period)
    freq = np.fft.fftshift(freq)+(fs/2)

    return freq, data_FFT


def plot_FFT(data):
    #Hente ut frekvens- og FFT-data fra func_FFT funksjonen
    freq, data_FFT = func_FFT(data[:, 0:2])
    
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Amplitude [V]")
    plt.title("FFT av ADC signaler")

    plt.plot(freq[1:-1], (np.abs(data_FFT[1:-1])/len(data)))

    plt.legend(('$ADC1$','$ADC2$','$ADC3$','$ADC4$','$ADC5$'), loc="upper right")
    plt.grid()
    plt.show()


def plot_periodogram(data):
    #Hente ut frekvens- og FFT-data fra func_FFT funksjonen
    freq, data_FFT = func_FFT(data[:, 0:2])

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relativ effekt, [dB]")
    plt.title("Periodogram av ADC-kanaler")

    plt.plot(freq[1:-1], 20*np.log10((abs(data_FFT[1:-1]))/max(abs(data_FFT[1:-2, 0]))))
    #add_window()

    plt.legend(('ADC1','ADC2','ADC3','ADC4','ADC5'), loc="upper right")
    plt.grid()
    plt.show()


def add_window():

    #Bartlett, blackman, hamming, hanningm, kaiser
    hanning = np.hanning(len(data))
    data_with_hanning = np.multiply(data, hanning[:, np.newaxis])

    #plot_FFT(data_with_hanning)
    plot_periodogram(data_with_hanning)


"""             ACOUSTICS LAB           """
def correlation(data):

    #Eks autokorrelasjon
    r_11 = ss.correlate(data[1:, 0:1], data[1:, 0:1])

    #Korrelasjon mellom alle sensorer
    r_12 = ss.correlate(data[1:, 0:1], data[1:, 1:2])
    r_23 = ss.correlate(data[1:, 1:2], data[1:, 2:3])
    r_13 = ss.correlate(data[1:, 0:1], data[1:, 2:3])

    t_r = ss.correlation_lags(len(data[1:, 0:1]), len(data[1:, 0:1]))

    return t_r, r_12, r_23, r_13, r_11

def n_values():

    t_r, r_12, r_23, r_13, r_11 = correlation(data)

    max_12 = np.argmax(r_12)
    max_23 = np.argmax(r_23)
    max_13 = np.argmax(r_13)

    n_12 = t_r[max_12]
    n_23 = t_r[max_23]
    n_13 = t_r[max_13]

    return n_12, n_23, n_13

def calc_angles():

    l12_max, l23_max, l13_max = n_values()

    y = (np.sqrt(3)*(l12_max+l13_max))
    x = (l12_max-l13_max-2*l23_max)

    arg = y/x

    print('Cosine: ' + str(np.cos(arg)))
    print('Sine: ' + str(np.sin(arg)))

    if (x) < 0:
        arg += np.pi

    return np.degrees(np.arctan2(y, x))


def plot_correlation(data):

    t = np.arange(0, dt*len(data), dt)
    t_r, r_12, r_23, r_13, r_11 = correlation(data)

    fig, ax = plt.subplots(2,1)

    ax[0].plot(t[1:], data[1:, 0:1], label = f'$x_{1}(t)$')
    ax[0].plot(t[1:], data[1:, 1:2], label = f'$x_{2}(t)$')
    ax[0].plot(t[1:], data[1:, 2:3], label = f'$x_{3}(t)$')
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Spenning [V]")

    #Autocorrelation
    #ax[1].plot(t_r, r_11, label = f'$r_{11}$')
    #Crosscorrelation
    ax[1].plot(t_r, r_12, label = f'$r_{12}$')
    ax[1].plot(t_r, r_23, label = f'$r_{23}$')
    ax[1].plot(t_r, r_13, label = f'$r_{13}$')
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Spenning [V]")

    plt.show()


#plot_ADC_channels(sample_period, data)
#plot_FFT(data)
#plot_periodogram(data)
#add_window()
#plot_correlation(data)
print('n-values: ' + str(n_values()))
print('Degree: ' + str(calc_angles()))

    