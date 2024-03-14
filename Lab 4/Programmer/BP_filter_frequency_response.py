### FILTER SCRIPT ###

import csv
import matplotlib.pyplot as plt

header = []
data = []


filename = "Network_BP_FILTER_1.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
print(data[0])
print(data[1])

time = [p[0] for p in data]
ch1 = [p[1] for p in data]
#ch2 = [p[3] for p in data]

#plt.plot(time,ch2, label='$y$(t)', color = 'orange')
plt.plot(time,ch1, label = '$v_o$(t)', color = 'blue')

plt.grid()
plt.xlabel('Frekvens [kHz]', fontsize=15)
plt.ylabel('Magnitude [dB]', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13) 

