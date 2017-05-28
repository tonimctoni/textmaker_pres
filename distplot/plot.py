import numpy as np
from matplotlib import pyplot as plt

def max_index(stuff):
    ret=0
    for i in range(len(stuff)):
        if stuff[i]>stuff[ret]: ret=i
    return ret

string="e latest sons who are finally called. my lord father cannot abide her now. i am a knight of the kingsguard. you know, davos s"
index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz"
substring=". i am a knight of the kingsguard."

with open("test0315c.txt") as f:
    content=map(float, f.read().split())

dists=[content[i*len(index_to_char):i*len(index_to_char)+len(index_to_char)] for i in range(len(content)/len(index_to_char))]

start_point=74
length=33
to_plot=list()
for i in range(length):
    to_plot.append(list())
    for j in range(4):
        mi=max_index(dists[i+start_point])
        to_plot[i].append((index_to_char[mi], dists[i+start_point][mi]))
        dists[i+start_point][mi]=0

c=0
for p in to_plot:
    barlist=plt.bar(np.multiply(np.arange(4), .5), np.array(map(lambda x: x[1], p)), .5)
    for i in range(4):
        if string[start_point+c]==p[i][0]: barlist[i].set_color('g')
    plt.xticks(np.multiply(np.arange(4), .5)+.25, np.array(map(lambda x: x[0].replace(" ", "_"), p)))
    plt.title(substring.replace(" ", "_")+"\n"+substring[:c+1].replace(" ", "_"))
    plt.axis([0, 2, 0, 1])
    # plt.grid(True)
    plt.savefig("%02i.png"%c)
    plt.clf()
    c+=1










