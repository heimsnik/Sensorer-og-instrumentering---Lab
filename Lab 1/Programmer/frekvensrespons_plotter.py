import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import FuncFormatter
import numpy as np 

def readCSV (path):
  print(path)
  data = []
  header = [] # removes first line of file
  filename = path
  print("yay")
  with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)
    #header = next(csvreader)  # removes first line of file

    for datapoint in csvreader:
        values = [float(value) for value in datapoint]
        data.append(values)
  return data



def magnitude(dataList, dataLabel):
  #Figure size (x,y) in inches. Move Legend if changed drasticly to avoid clipping
  fig = plt.figure(1, figsize=(14.5, 6.5))
  plt.rc('font', size=18)  
  
  #number of rows/cols of subplots 
  ax = fig.add_subplot(1, 1, 1)
  
  #max num ticks in axis
  max_yticks = 15
  max_xticks = 20 #irrelevant due to log scale
  yloc = plt.MaxNLocator(max_yticks)
  xloc = plt.MaxNLocator(max_xticks)
  ax.yaxis.set_major_locator(yloc)
  ax.xaxis.set_major_locator(xloc)

  #Use log scale  
  ax.set_xscale('log')
  # ax.set_yscale('log')

  #Plot data
  for i in range (0,len(dataList)):
    time = [p[0] for p in dataList[i]]
    measurement = [p[1] for p in dataList[i]]
    plt.plot(time, measurement, "-")

  #Add lines for 3 dB
  plt.hlines([-3,-3],2,26.76,'grey','dashed')
  plt.vlines([26.76,26.76],-25,-3,'grey','dashed')
  plt.text(27.5,-26.5,"$f_c$",fontsize = 20) #26.7 Hz
  #plt.hlines([-3,-3],2,26.76,'green','dashed')
  plt.vlines([23.2,23.2],-25,-1.55,'grey','dashed')
  plt.vlines([13.89,13.89],-25,1.85,'grey','dashed')
  plt.hlines([-28,-28],23.2,26.76,'red','solid')
  plt.vlines([23.2,26.76],-26,-30,'red','solid')
  plt.text(23,-32,"$\Delta f_c$",fontsize = 20)
  plt.text(20.5,-26.5,"$f_c'$",fontsize = 20)
  plt.text(12.5,-26.5,"$f_r$",fontsize = 20)

  #labels  
  plt.xlabel("Frekvens [Hz]",fontsize = 18)
  plt.ylabel("Relativ amplitude [dB]",fontsize = 18)

  
  #legend. Source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
  plt.legend(dataLabel,fontsize = 16)#, bbox_to_anchor=(0.5, -0.12), loc="upper center",fancybox=True, ncol=8, borderaxespad=0))
  plt.tight_layout(rect=[0,0,1,0.98])

  #Final touch
  plt.title("Frekvensrespons til Pi-filter",fontsize = 18)
  plt.grid(True)
  plt.show()

def phase(dataList, dataLabel):
  #Figure size (x,y) in inches. Move Legend if changed drasticly
  fig = plt.figure(1, figsize=(14.5, 6.5))
  plt.rc('font', size=16)      # fontsize of the tick labels
  
  #number of rows/cols of subplots 
  ax = fig.add_subplot(1, 1, 1)
  
  #max num ticks in axis
  max_yticks = 15
  max_xticks = 20 #irrelevant due to log scale
  yloc = plt.MaxNLocator(max_yticks)
  xloc = plt.MaxNLocator(max_xticks)
  ax.yaxis.set_major_locator(yloc)
  ax.xaxis.set_major_locator(xloc)

  #Use log scale  
  ax.set_xscale('log')
  # ax.set_yscale('log')

  #plot data
  for i in range (0,len(dataList)):
    time = [p[0] for p in dataList[i]]
    measurement = [p[3] for p in dataList[i]]
    plt.plot(time, measurement, "-")

  #labels  
  plt.xlabel("Frekvens [Hz]",fontsize = 16)
  plt.ylabel("Fase [grader]",fontsize = 16)
  
  #Legend
  #plt.legend(dataLabel)#, bbox_to_anchor=(0.5, -0.12), loc="upper center",fancybox=True, ncol=8, borderaxespad=0)
  plt.tight_layout(rect=[0,0,1,0.98])
  
  
  #Final touch
  plt.grid(True)
  plt.title('Faserespons',fontsize = 16)
  plt.show()


def bodeDiagram(fileList,dataLabel):
  if len(fileList)!=len(dataLabel):
    print("\n\nMissing labels for grafs\n\n")
  dataList= []
  for i in range (0,len(fileList)):
    dataList.append(readCSV(fileList[i]))
  magnitude(dataList, dataLabel)
  phase(dataList, dataLabel)

files = ["network_data.csv"]
dataLabel = ["Amplituderespons"]
bodeDiagram(files, dataLabel)