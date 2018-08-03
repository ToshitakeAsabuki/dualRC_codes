# coding: UTF-8
from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.matlib
import os
import shutil
import sys
import matplotlib.cm as cm

def f(x):
    ans =np.tanh(x/3)
    if ans < 0:
        ans = 0
    return ans

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': [10 / 2.54, 6 / 2.54]}

sigma = 0.3

input_gain = 2
minimum_length = 5

chunk = ['a', 'b', 'c', 'd']
random_elements = ['e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
elements_list = chunk + random_elements

nIn2Rec = len(elements_list)
g_FG = 1
width = 50
N = 300
n = 300
p = 1
g = 1.5

input_shape = np.zeros(width*2)

input_shape[0:width] = input_gain*(1 - np.exp(-(np.arange(0,width)/10)))
input_shape[width:2*width] = input_gain*np.exp(-(np.arange(0,width)/10))

alpha = 100
synapse_list1 = np.random.choice(np.arange(N), n, replace=False)
synapse_list2 = np.random.choice(np.arange(N), n, replace=False)
nsecs = width * 10000
dt = 1
tau = 10
learn_every = 2
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)
scale = 1.0 / np.sqrt(p * N)

symbol_list = np.zeros(simtime_len)

M = np.zeros((N, N))
M2 = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if np.random.rand() < p:
            M[i, j] = (np.random.randn()) * g * scale
        if np.random.rand() < p:
            M2[i, j] = (np.random.randn()) * g * scale

nRec2Out = N

wo1 = np.random.randn(n) / np.sqrt(n)

wo2 = np.random.randn(n) / np.sqrt(n)

wf1 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf2 = (np.random.rand(N) - 0.5) * 2 * g_FG

win = np.zeros((N, nIn2Rec))
win2 = np.zeros((N, nIn2Rec))

for i in range(N):
    win[i, np.random.randint(0, nIn2Rec)] = np.random.randn()
    win2[i, np.random.randint(0, nIn2Rec)] = np.random.randn()

x01 = 0.5 * np.random.randn(N)
x02 = 0.5 * np.random.randn(N)
z01 = 0.5 * np.random.randn()
z02 = 0.5 * np.random.randn()

P = (1.0 / alpha) * np.eye(n)
P2 = (1.0 / alpha) * np.eye(n)

I = np.zeros((nIn2Rec, simtime_len))

target_list = np.zeros(simtime_len)

random_seq_len = np.random.randint(minimum_length , minimum_length + 4)
random_seq = [''] * random_seq_len
for i in range(random_seq_len):
    random_seq[i] = random_elements[np.random.randint(0, len(random_elements))]
input_type = random_seq
m = 0


for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if input_type == chunk:
            if m == len(chunk) - 1:
                random_seq_len = np.random.randint(minimum_length , minimum_length + 4)
                random_seq = [''] * random_seq_len
                for l in range(random_seq_len):
                    random_seq[l] = random_elements[np.random.randint(0, len(random_elements))]

                input_type = random_seq
                m = 0

            else:
                input_type = chunk
                m += 1

        elif input_type == random_seq:
            if m == len(random_seq) - 1:
                input_type = chunk
                m = 0

            else:
                input_type = random_seq
                m += 1

        I[elements_list.index(input_type[m]), i:min(i + width * 2, simtime_len)] = input_shape[0:min(i + width * 2, simtime_len) - i]

print("")
print("***********")
print("Learning... ")
print("***********")

window = 300

z1_list = np.zeros(window * width)
z2_list = np.zeros(window * width)

x = x01
x2 = x02
r = np.tanh(x)
r2 = np.tanh(x2)

z1 = z01
z2 = z02

for i in range(simtime_len):

    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")
    x = (1.0 - dt / tau) * x + np.dot(M, r) * dt / tau + np.dot(win, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf1 * z1 * dt / tau
    x2 = (1.0 - dt / tau) * x2 + np.dot(M2, r2) * dt / tau + np.dot(win2, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf2 * z2 * dt / tau

    r = np.tanh(x)
    r2 = np.tanh(x2)

    z1 = np.dot(wo1, r[synapse_list1])
    z2 = np.dot(wo2, r2[synapse_list2])

    z1_list = np.roll(z1_list, -1)
    z1_list[-1] = z1
    z2_list = np.roll(z2_list, -1)
    z2_list[-1] = z2

    y1 = (z2 - np.mean(z2_list)) / np.std(z2_list)
    y2 = (z1 - np.mean(z1_list)) / np.std(z1_list)

    if (i + 1) % learn_every == 0:

        if i > width * window:

            k = np.dot(P, r[synapse_list1])
            rPr = np.dot(r[synapse_list1], k)
            c = 1.0 / (1.0 + rPr)
            P = P - c * np.outer(k, k)

            k2 = np.dot(P2, r2[synapse_list2])
            rPr2 = np.dot(r2[synapse_list2], k2)
            c2 = 1.0 / (1.0 + rPr2)
            P2 = P2 - c2 * np.outer(k2, k2)

            e1 = z1 - f(y1)
            e2 = z2 - f(y2)
            dw1 = -c * e1 * k
            dw2 = -c2 * e2 * k2

            wo1 = wo1 + dw1
            wo2 = wo2 + dw2


print("")
print("***********")
print("Testing... ")
print("***********")
test_len = 3000
I = np.zeros((nIn2Rec, test_len))
random_seq_len = np.random.randint(minimum_length , minimum_length + 4)
random_seq = [''] * random_seq_len
for i in range(random_seq_len):
    random_seq[i] = random_elements[np.random.randint(0, len(random_elements))]
input_type= random_seq
m = 0
chunk_start = []
for i in range(test_len):
    if (i % width == 0 and i > 0):
        if input_type == chunk:
            if m == len(chunk) - 1:
                random_seq_len = np.random.randint(minimum_length , minimum_length + 4)
                random_seq = [''] * random_seq_len
                for l in range(random_seq_len):
                    random_seq[l] = random_elements[np.random.randint(0, len(random_elements))]

                input_type = random_seq
                m = 0

            else:
                input_type = chunk
                m += 1

        elif input_type == random_seq:
            if m == len(random_seq) - 1:
                input_type = chunk
                m = 0

                chunk_start.append(i)

            else:
                input_type = random_seq
                m += 1

        I[elements_list.index(input_type[m]), i:min(i + width * 2, test_len)] = input_shape[0:min(i + width * 2, test_len) - i]

z1_list = np.zeros(test_len)
z2_list = np.zeros(test_len)

x = x01
x2 = x02
r = np.tanh(x)
r2 = np.tanh(x2)

z1 = z01
z2 = z02

for i in range(test_len):
    x = (1.0 - dt / tau) * x + np.dot(M, r) * dt / tau + np.dot(win, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf1 * z1 * dt / tau
    x2 = (1.0 - dt / tau) * x2 + np.dot(M2, r2) * dt / tau + np.dot(win2, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf2 * z2 * dt / tau

    r = np.tanh(x)
    r2 = np.tanh(x2)

    z1 = np.dot(wo1, r[synapse_list1])
    z2 = np.dot(wo2, r2[synapse_list2])

    z1_list[i] = z1
    z2_list[i] = z2

fig = plt.figure(figsize=(7, 2))
ax = plt.subplot(1, 1, 1)
pl.plot(z1_list, lw=1.5, c='steelblue')

for i in (chunk_start):
    pl.axvspan(i, i + width * len(chunk), facecolor='steelblue', alpha=0.3,linewidth=0)

pl.ylim(np.min(z1_list[0:test_len]) - 0.1,
        np.max(z1_list[0:test_len]) + 0.1)

plt.xlabel("Time [ms]", fontsize=11)
plt.ylabel("Activity", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.15)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('readout_double.pdf', fmt='pdf', dpi=350)
