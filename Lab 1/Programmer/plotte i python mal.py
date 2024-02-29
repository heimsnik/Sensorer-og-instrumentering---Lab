import csv
import matplotlib.pyplot as plt

header = []
data = []


filename = "C:/Users/cmhei/OneDrive/Dokumenter/Semester 6/TTK4280 Sensorer og instrumentering/Lab/Lab 1/Programmer/scope_noice.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
print(data[0])
print(data[1])
#fig = plt.figure(1, figsize=(14.5, 6.5))
#ax = fig.add_subplot(1, 1, 1)
#ax.hlines([5.04,5.04],300,650, linestyles='solid', colors='red')
#ax.vlines([466,478],5.3,4, linestyles='dashed', colors='red')
#ax.text(570,5.25,"$\Delta$ $t$ = 12 $\mu s$",fontsize = 18)
time = [p[0] for p in data]
shiftedtime = []
for i in range(len(time)):
    shiftedtime.append(time[i]+0.0513)
ch1 = [p[1] for p in data]
ch2 = [p[2] for p in data]
#ch3 = [(p[3]) for p in data]
#ch4 = [(p[4]*4.8-0.2) for p in data]

plt.rc('font', size=12)  
plt.xlabel("Tid [s]",fontsize = 12)
plt.ylabel("Spenning [V]",fontsize = 12)

plt.title("Støy i forsyningsspenninger")
plt.grid(True)
fig = plt.plot(shiftedtime,ch2, label = "$VDDA(t)$")
fig = plt.plot(shiftedtime,ch1, label = "$VDDD(t)$")
#fig = plt.plot(time,ch4, label = "$v_Q$ med $R_L$ = 100$\Omega$")
#fig = plt.plot(time,ch3, label = "$v_Q$ med $R_L$ = 100$\Omega$")

plt.legend()

plt.show()

    
