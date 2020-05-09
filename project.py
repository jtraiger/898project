#!/usr/bin/env python
# coding: utf-8

# In[197]:


import pandas
import numpy as np

# Get age information from demographic information
dem = pandas.read_csv("demographics.txt"," ").to_numpy()
names_full = [dem[i,0] for i in range(0,dem.shape[0])]
ages_full = [int('18' in dem[i,2]) for i in range(0,dem.shape[0])]

# Create dictionary of names and ages
full_D = dict(zip(names_full, ages_full))
print(full_D)


# In[198]:


# Create average typing speed dictionary  - not really used
data = pandas.read_csv("DSL-StrongPasswordData.csv").to_numpy()
speeds = [(data[i,1:].sum()) for i in range(0,data.shape[0])]
num_trials = 400
avgs = [np.mean(speeds[400*i:400*(i+1)-1]) for i in range(int(len(speeds)/num_trials))]
names = [data[400*i,0] for i in range(int(len(speeds)/num_trials))]
D = dict(zip(names, avgs))


# In[199]:


# Show difference exists between younger and older ppl in terms of typing speed
y = []
o = []
for n in names: 
    avg = D[n]
    if(full_D[n] == 0): o = o + [avg]
    else: y = y + [avg]
        
print(np.mean(y))
print(np.mean(o))


# In[200]:


#Try to create a graph
from scipy.stats.stats import pearsonr
import math
import matplotlib.pyplot as plt
import networkx as nx  # importing networkx package

data = pandas.read_csv("DSL-StrongPasswordData.csv").to_numpy()
#avgs = np.array(data[0*400:(0+1)*400,1:].mean(axis=0))
# Use 8th trial as it is the most stable
avgs = np.array(data[8,1:])
for i in range(1,int(data.shape[0]/400)): 
    #avgdata =  data[i*400:(i+1)*400,1:].mean(axis=0)
    avgdata =  data[i*400 + 8, 1:]
    avgs = np.vstack((avgs,np.array(avgdata)))


# In[201]:


# Graph based on pearson scores pairwise between typists
# typist 42 is removed because they are an extreme outlier
corrs = np.zeros((51,51))
for i in range(51):
    for j in range(51):
        corrs[i,j] = pearsonr(avgs[i],avgs[j])[0]
mint = np.min(corrs)
corrs = corrs - np.diag([1]*51)
maxt = np.max(corrs)

corrs = (corrs - mint)/(maxt-mint)
for i in range(51):
    corrs[i,i] = 0
print(corrs)
print(np.mean(corrs))
corr2 = (corrs>np.mean(corrs))
derp = False
for i in range(51):
    for j in range(51):
        corr2[i,j] = int(corr2[i,j])
        if(corr2[i,j] == 1 ): derp = True
    if(derp == False): print(i)
    derp = False
#plt.matshow(corr2)


A = corr2
A2 = np.array(A)
A2 = np.delete(A2, 42,0)
A2 = np.delete(A2,42,1)

G = nx.from_numpy_matrix(A2)
nx.draw(G)


# In[204]:


# Graph based on euclidean distance between feature vectors of pairwise users
corrs = np.zeros((51,51))
for i in range(51):
    for j in range(51):
        corrs[i,j] = np.sum((np.array(avgs[i]) - np.array(avgs[j]))**2)

mint = np.min(corrs)
corrs = corrs - np.diag([1]*51)
maxt = np.max(corrs)

corrs = (corrs - mint)/(maxt-mint)
for i in range(51):
    corrs[i,i] = 0

corr2 = (corrs>np.mean(corrs))


G = nx.from_numpy_matrix(corr2)
nx.draw(G)


# In[205]:


# graph based on differences between average typing speed
corrs = np.zeros((51,51))
for i in range(51):
    for j in range(51):
        corrs[i,j] = np.abs(np.sum(np.array(avgs[i]))/len(avgs[i]) - np.sum(np.array(avgs[j]))/len(avgs[i]))

mint = np.min(corrs)
corrs = corrs - np.diag([1]*51)
maxt = np.max(corrs)

corrs = (corrs - mint)/(maxt-mint)
for i in range(51):
    corrs[i,i] = 0

corr2 = (corrs>np.mean(corrs))
A = corr2

G = nx.from_numpy_matrix(A)
nx.draw(G)

