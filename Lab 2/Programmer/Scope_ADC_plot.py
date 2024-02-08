import csv
import matplotlib.pyplot as plt

header = []
data = []


filename = "Scope_ADC_1_2.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
#print(data[0])
#print(data[1])

time = [p[0] for p in data]
ch1 = [p[1] for p in data]
ch2 = [p[2] for p in data]

plt.plot(time,ch2, label='$s$(t)', color = 'orange')
plt.plot(time,ch1, label = '$x$(t)', color = 'red')


plt.grid()
plt.xlabel('Tid [s]', fontsize=15)
plt.ylabel('Spenning [V]', fontsize=15)
plt.legend(loc='upper right', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=15) 
