import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt

samples = 1000
start = 0
shift_1 = 0.2
shift_2 = 0.5
stop = 1
A = 5
l = 0.05

fs = 10000

#two signals
t = np.linspace(start,stop,fs*(stop-start))

x_1 = A*np.exp(-((t-shift_1)/l)**2)
x_2 = A*np.exp(-((t-shift_2)/l)**2)

correlation = ss.correlate(x_1,x_2)
t_c = ss.correlation_lags(len(x_1),len(x_2))
fig, ax = plt.subplots(2,1)
ax[0].plot(t,x_1,label = "x1")
ax[0].plot(t,x_2,label = "x2")
ax[0].legend()
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Amplitude [V]")

ax[1].plot(t_c/fs,correlation)
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Amplitude [V]")

plt.show()