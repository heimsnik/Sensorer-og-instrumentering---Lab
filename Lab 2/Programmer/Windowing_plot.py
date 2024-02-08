
import numpy as np
import matplotlib.pyplot as plt

A = 1
n = 100

t = np.arange(-4*np.pi, 14*np.pi, 0.2)
x = A * np.cos(t)

t_before = np.where((t >= -4*np.pi) & (t <= 0), t, np.nan)
t_filtered = np.where((t >= 0) & (t <= 10*np.pi), t, np.nan)
t_after = np.where((t >= 10*np.pi) & (t <= 14*np.pi), t, np.nan)
W = []
W = [1 for i in range(len(t_filtered))]
x_before = []
x_after = []
x_before = [0 for i in range(len(t_before))]
x_after = [0 for i in range(len(t_after))]



#plt.stem(t_filtered, W, linefmt='g-', markerfmt='go', basefmt='k-', label='w[n])
plt.stem(t_before, x_before, linefmt='b-', markerfmt='bo', basefmt='k-', label=None)
plt.stem(t_after, x_after, linefmt='b-', markerfmt='bo', basefmt='k-', label=None)

plt.xlabel("t [s]", fontsize = 15)
plt.ylabel("Amplitude [V]", fontsize = 15)
#plt.title("Signal x(t)")
#plt.stem(t, x, markerfmt='bo', basefmt='k-', label="x(t)")#,  use_line_collection=True, markerfmt='o', basefmt='k-')

t_filtered = np.where((t >= 0) & (t <= 10*np.pi), t, np.nan)
x_filtered = A * np.cos(t_filtered)

plt.stem(t_filtered, x_filtered, linefmt='b-', markerfmt='bo', basefmt='k-', label="x(t)")#,  use_line_collection=True, markerfmt='o', basefmt='k-')

plt.legend(loc="upper right", fontsize = 15)

plt.grid()
plt.show()

