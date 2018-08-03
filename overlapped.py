# coding: UTF-8
from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.matlib
import os
import shutil


def f(x):
    ans = max(np.tanh(x/3), 0)

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


sigma1 = 0.1
sigma2 = 0.1

input_gain = 2
interval = 3
chunk1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
chunk2 = ['i', 'j', 'k', 'd', 'e', 'l', 'm', 'n']
random_elements = ['o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
elements_list = chunk1 + ['i', 'j', 'k', 'l', 'm', 'n'] + random_elements

nIn2Rec = len(elements_list)
g_FG = 1
width = 50
N = 500
n = 300
p = 1
g = 1.5
lateral = 1
input_shape = np.zeros(width*2)
for i in range(width):
    input_shape[i] = input_gain*(1 - np.exp(-(i/10)))
    input_shape[width+i] = input_gain*np.exp(-(i/10))
alpha = 100
synapse_list1 = np.random.choice(np.arange(N), n, replace=False)
synapse_list2 = np.random.choice(np.arange(N), n, replace=False)
nsecs = width *500000
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

for i in range(N):
    for j in range(N):
        if np.random.rand() < p:
            M2[i, j] = (np.random.randn()) * g * scale

nRec2Out = N

wo1 = np.random.randn(n) / np.sqrt(n)

wo2 = np.random.randn(n) / np.sqrt(n)

wo3 = np.random.randn(n) / np.sqrt(n)

wo4 = np.random.randn(n) / np.sqrt(n)

wf1 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf2 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf3 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf4 = (np.random.rand(N) - 0.5) * 2 * g_FG

win = np.zeros((N, nIn2Rec))
win2 = np.zeros((N, nIn2Rec))

for i in range(N):
    win[i, np.random.randint(0, nIn2Rec)] = np.random.randn()

for i in range(N):
    win2[i, np.random.randint(0, nIn2Rec)] = np.random.randn()

x01 = 0.5 * np.random.randn(N)
x02 = 0.5 * np.random.randn(N)

z01 = 0.5 * np.random.randn()
z02 = 0.5 * np.random.randn()
z03 = 0.5 * np.random.randn()
z04 = 0.5 * np.random.randn()

P = (1.0 / alpha) * np.eye(n)
P2 = (1.0 / alpha) * np.eye(n)

I = np.zeros((nIn2Rec, simtime_len))

rand_len = np.random.randint(interval + 2, interval + 10)
rand = [''] * rand_len
for i in range(rand_len):
    rand[i] = random_elements[np.random.randint(0, len(random_elements))]
chunk = rand
m = 0

for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if chunk == chunk1:
            if m == len(chunk1) - 1:
                rand_len = np.random.randint(interval + 2, interval + 10)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0

            else:
                chunk = chunk1
                m += 1

        elif chunk == chunk2:
            if m == len(chunk1) - 1:
                rand_len = np.random.randint(interval + 2, interval + 10)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0
            else:
                chunk = chunk2
                m += 1

        elif chunk == rand:
            if m == len(rand) - 1:
                dice = np.random.rand()
                if dice < 1/2:
                    chunk = chunk1
                    m = 0

                else:
                    chunk = chunk2
                    m = 0


            else:
                chunk = rand
                m += 1
 
        input_end = min(i + width * 2, simtime_len)
        I[elements_list.index(chunk[m]), i:input_end] = input_shape[0:input_end - i]

print("")
print("***********")
print("Learning... ")
print("***********")

window = 300

z1_list = np.zeros(window * width)
z2_list = np.zeros(window * width)
z3_list = np.zeros(window * width)
z4_list = np.zeros(window * width)

w_list = []
dw_list = []
x = x01
x2 = x02
r = np.tanh(x)
r2 = np.tanh(x2)

z1 = z01
z2 = z02
z3 = z03
z4 = z04

for i in range(simtime_len):

    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")
    x = (1.0 - dt / tau) * x + np.dot(M, r) * dt / tau + np.dot(win, I[:, i]) * dt / tau + sigma1 * np.sqrt(
        dt) * np.random.randn(N) + wf1 * z1 * dt / tau + wf2 * z2 * dt / tau
    x2 = (1.0 - dt / tau) * x2 + np.dot(M2, r2) * dt / tau + np.dot(win2, I[:, i]) * dt / tau + sigma2 * np.sqrt(
        dt) * np.random.randn(N) + wf3 * z3 * dt / tau + wf4 * z4 * dt / tau

    r = np.tanh(x)
    r2 = np.tanh(x2)

    z1 = np.dot(wo1, r[synapse_list1])
    z2 = np.dot(wo2, r[synapse_list1])
    z3 = np.dot(wo3, r2[synapse_list2])
    z4 = np.dot(wo4, r2[synapse_list2])

    z1_list = np.roll(z1_list, -1)
    z1_list[-1] = z1
    z2_list = np.roll(z2_list, -1)
    z2_list[-1] = z2
    z3_list = np.roll(z3_list, -1)
    z3_list[-1] = z3
    z4_list = np.roll(z4_list, -1)
    z4_list[-1] = z4

    y1 = (z3 - np.mean(z3_list)) / np.std(z3_list) - lateral * (z4 - np.mean(z4_list)) / np.std(z4_list)
    y2 = (z4 - np.mean(z4_list)) / np.std(z4_list) - lateral * (z3 - np.mean(z3_list)) / np.std(z3_list)
    y3 = (z1 - np.mean(z1_list)) / np.std(z1_list) - lateral * (z2 - np.mean(z2_list)) / np.std(z2_list)
    y4 = (z2 - np.mean(z2_list)) / np.std(z2_list) - lateral * (z1 - np.mean(z1_list)) / np.std(z1_list)
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
            e3 = z3 - f(y3)
            e4 = z4 - f(y4)
            dw1 = -c * e1 * k
            dw2 = -c * e2 * k
            dw3 = -c2 * e3 * k2
            dw4 = -c2 * e4 * k2

            wo1 = wo1 + dw1
            wo2 = wo2 + dw2
            wo3 = wo3 + dw3
            wo4 = wo4 + dw4


print("")
print("***********")
print("Testing... ")
print("***********")
I = np.zeros((nIn2Rec, simtime_len))
chunk_list = np.zeros(simtime_len)
chunk2_list = np.zeros(simtime_len) - 1
rand_len = np.random.randint(interval + 2, interval + 10)
rand = [''] * rand_len
for i in range(rand_len):
    rand[i] = random_elements[np.random.randint(0, len(random_elements))]
chunk = rand
m = 0
chunk1_start = []
chunk2_start = []
for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if chunk == chunk1:
            if m == len(chunk1) - 1:
                rand_len = np.random.randint(interval + 2, interval + 10)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0

            else:
                chunk = chunk1
                m += 1

        elif chunk == chunk2:
            if m == len(chunk1) - 1:
                rand_len = np.random.randint(interval + 2, interval + 10)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0
            else:
                chunk = chunk2
                m += 1

        elif chunk == rand:
            if m == len(rand) - 1:
                dice = np.random.rand()
                if dice < 1/2:
                    chunk = chunk1
                    m = 0
                    chunk_list[i:min(i + width * len(chunk1), simtime_len)] = 1
                    chunk1_start.append(i)
                else:
                    chunk = chunk2
                    m = 0
                    chunk2_list[i:min(i + width * len(chunk1), simtime_len)] = 1
                    chunk2_start.append(i)

            else:
                chunk = rand
                m += 1
    
        input_end = min(i + width * 2, simtime_len)
        I[elements_list.index(chunk[m]), i:input_end] = input_shape[0:input_end - i]
test_len = min(width * 2000, simtime_len)
z1_list = np.zeros(test_len)
z2_list = np.zeros(test_len)

chunk_list2 = np.zeros(test_len)
x = x01
x2 = x02
r = np.tanh(x)
r2 = np.tanh(x2)

z1 = z01
z2 = z02
z3 = z03
z4 = z04

for i in range(test_len):

    x = (1.0 - dt / tau) * x + np.dot(M, r) * dt / tau + np.dot(win, I[:, i]) * dt / tau + sigma1 * np.sqrt(
        dt) * np.random.randn(N) + wf1 * z1 * dt / tau + wf2 * z2 * dt / tau
    x2 = (1.0 - dt / tau) * x2 + np.dot(M2, r2) * dt / tau + np.dot(win2, I[:, i]) * dt / tau + sigma2 * np.sqrt(
        dt) * np.random.randn(N) + wf3 * z3 * dt / tau+ wf4 * z4 * dt / tau

    r = np.tanh(x)
    r2 = np.tanh(x2)

    z1 = np.dot(wo1, r[synapse_list1])
    z2 = np.dot(wo2, r[synapse_list1])
    z3 = np.dot(wo3, r2[synapse_list2])
    z4 = np.dot(wo4, r2[synapse_list2])

    z1_list[i] = z1
    z2_list[i] = z2

chunk1test_start = [x for x in chunk1_start if x < test_len]

z1_avg_chunk1 = np.zeros(width * (len(chunk1) + interval * 2))
z1_avg_chunk2 = np.zeros(width * (len(chunk1) + interval * 2))
chunk2test_start = [x for x in chunk2_start if x < test_len]
z2_avg_chunk1 = np.zeros(width * (len(chunk1) + interval * 2))
z2_avg_chunk2 = np.zeros(width * (len(chunk1) + interval * 2))

for j in chunk1test_start[1:-5]:
    z1_avg_chunk1 += z1_list[j - width * interval:j + width * len(chunk1) + interval * width]
    z2_avg_chunk1 += z2_list[j - width * interval:j + width * len(chunk1) + interval * width]

for j in chunk2test_start[1:-5]:
    z1_avg_chunk2 += z1_list[j - width * interval:j + width * len(chunk1) + interval * width]
    z2_avg_chunk2 += z2_list[j - width * interval:j + width * len(chunk1) + interval * width]
z1_avg_chunk1 /= len(chunk1test_start[1:-5])
z2_avg_chunk1 /= len(chunk1test_start[1:-5])
z1_avg_chunk2 /= len(chunk2test_start[1:-5])
z2_avg_chunk2 /= len(chunk2test_start[1:-5])

avg_chunk_range = np.zeros(width * (len(chunk1) + interval * 4)) - 1
avg_chunk_range[interval * width:width * len(chunk1) + interval * width] = 1
fig = plt.figure(figsize=(7, 2))
ax = plt.subplot(1, 1, 1)
pl.plot(z1_list, lw=2.5, c='orangered')
pl.plot(z2_list, lw=2.5, c='dodgerblue')


for i in (chunk1test_start):
    pl.axvspan(i, i + width * len(chunk1), facecolor='dodgerblue', alpha=0.3,linewidth=0)
for i in (chunk2test_start):
    pl.axvspan(i, i + width * len(chunk2), facecolor='orangered', alpha=0.3,linewidth=0)

pl.ylim(min(np.min(z1_list[0:test_len]), np.min(z2_list[0:test_len])) - 0.1,
            max(np.max(z1_list[0:test_len]), np.max(z2_list[0:test_len])) + 0.1)
plot_start = 0
pl.xlim(plot_start, plot_start + 6000)


plt.xlabel("Time [ms]", fontsize=15)
plt.ylabel("Activity", fontsize=15)

fig.subplots_adjust(bottom=0.25, left=0.15)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('readouts.pdf', fmt='pdf', dpi=350)
