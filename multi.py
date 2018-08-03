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
    ans = np.tanh(x/3)
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

sigma = 0.2

input_gain = 2
interval = 3
chunk1 = ['a', 'b', 'c', 'd']
chunk2 = ['e', 'f', 'g', 'h']
chunk3 = ['i', 'j', 'k', 'l']
random_elements = ['m', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
elements_list = chunk1 + chunk2 + chunk3 + random_elements

nIn2Rec = len(elements_list)
g_FG = 1
width = 50
N = 600
n = 300
p = 0.5
g = 1.5
lateral = 0.5
input_shape = np.zeros(width*2)
for i in range(width):
    input_shape[i] = input_gain*(1 - np.exp(-(i/10)))
    input_shape[width+i] = input_gain*np.exp(-(i/10))
alpha = 100

nsecs = width * 10000
dt = 1
tau = 10
learn_every = 2
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)
scale = 1.0 / np.sqrt(p * N)
synapse_list1 = np.random.choice(np.arange(N), n, replace=False)
synapse_list2 = np.random.choice(np.arange(N), n, replace=False)

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

wo5 = np.random.randn(n) / np.sqrt(n)

wo6 = np.random.randn(n) / np.sqrt(n)

wf1 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf2 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf3 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf4 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf5 = (np.random.rand(N) - 0.5) * 2 * g_FG
wf6 = (np.random.rand(N) - 0.5) * 2 * g_FG

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
z05 = 0.5 * np.random.randn()
z06 = 0.5 * np.random.randn()

P = (1.0 / alpha) * np.eye(n)
P2 = (1.0 / alpha) * np.eye(n)

I = np.zeros((nIn2Rec, simtime_len))


rand_len = np.random.randint(interval + 2, interval + 5)
rand = [''] * rand_len
for i in range(rand_len):
    rand[i] = random_elements[np.random.randint(0, len(random_elements))]
chunk = rand
m = 0

for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if chunk == chunk1:
            if m == len(chunk1) - 1:
                rand_len = np.random.randint(interval + 2, interval + 5)
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
                rand_len = np.random.randint(interval + 2, interval + 5)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0
            else:
                chunk = chunk2
                m += 1

        elif chunk == chunk3:
            if m == len(chunk3) - 1:
                rand_len = np.random.randint(interval + 2, interval + 5)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0
            else:
                chunk = chunk3
                m += 1

        elif chunk == rand:
            if m == len(rand) - 1:
                dice = np.random.rand()
                if dice < 1/3:
                    chunk = chunk1
                    m = 0

                elif dice < 2/3:
                    chunk = chunk2
                    m = 0

                else:
                    chunk = chunk3
                    m = 0

            else:
                chunk = rand
                m += 1
        # print(chunk[m])
        input_end = min(i + width * 2, simtime_len)
        I[elements_list.index(chunk[m]), i:input_end] = input_shape[0:input_end - i]

print("")
print("***********")
print("Learning... ")
print("***********")

window = 300
count = 0
z1_list = np.zeros(window * width)
z2_list = np.zeros(window * width)
z3_list = np.zeros(window * width)
z4_list = np.zeros(window * width)
z5_list = np.zeros(window * width)
z6_list = np.zeros(window * width)


x = x01
x2 = x02
r = np.tanh(x)
r2 = np.tanh(x2)

z1 = z01
z2 = z02
z3 = z03
z4 = z04
z5 = z05
z6 = z06

for i in range(simtime_len):

    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")


    x = (1.0 - dt / tau) * x + np.dot(M, r) * dt / tau + np.dot(win, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf1 * z1 * dt / tau + wf2 * z2 * dt / tau + wf3 * z3 * dt / tau
    x2 = (1.0 - dt / tau) * x2 + np.dot(M2, r2) * dt / tau + np.dot(win2, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf4 * z4 * dt / tau + wf5 * z5 * dt / tau + wf6 * z6 * dt / tau

    r = np.tanh(x)
    r2 = np.tanh(x2)

    z1 = np.dot(wo1, r[synapse_list1])
    z2 = np.dot(wo2, r[synapse_list1])
    z3 = np.dot(wo3, r[synapse_list1])
    z4 = np.dot(wo4, r2[synapse_list2])
    z5 = np.dot(wo5, r2[synapse_list2])
    z6 = np.dot(wo6, r2[synapse_list2])



    z1_list = np.roll(z1_list, -1)
    z1_list[-1] = z1
    z2_list = np.roll(z2_list, -1)
    z2_list[-1] = z2
    z3_list = np.roll(z3_list, -1)
    z3_list[-1] = z3
    z4_list = np.roll(z4_list, -1)
    z4_list[-1] = z4
    z5_list = np.roll(z5_list, -1)
    z5_list[-1] = z5
    z6_list = np.roll(z6_list, -1)
    z6_list[-1] = z6

    z1_norm=(z1 - np.mean(z1_list)) / np.std(z1_list)
    z2_norm=(z2 - np.mean(z2_list)) / np.std(z2_list)
    z3_norm=(z3 - np.mean(z3_list)) / np.std(z3_list)
    z4_norm=(z4 - np.mean(z4_list)) / np.std(z4_list)
    z5_norm=(z5 - np.mean(z5_list)) / np.std(z5_list)
    z6_norm=(z6 - np.mean(z6_list)) / np.std(z6_list)

    y1 = z4_norm - lateral * (z5_norm+z6_norm)
    y2 =z5_norm - lateral * (z4_norm+z6_norm)
    y3 =z6_norm - lateral * (z4_norm+z5_norm)
    y4 = z1_norm - lateral * (z2_norm+z3_norm)
    y5 = z2_norm - lateral * (z1_norm+z3_norm)
    y6 = z3_norm - lateral * (z1_norm+z2_norm)
    if (i + 1) % learn_every == 0:

        if i > width * window:

            k = np.dot(P, r[synapse_list1])
            rPr = np.dot(r[synapse_list1], k)
            c = 1.0 / (1.0 + rPr)
            P -= c * np.outer(k, k)

            k2 = np.dot(P2, r2[synapse_list2])
            rPr2 = np.dot(r2[synapse_list2], k2)
            c2 = 1.0 / (1.0 + rPr2)
            P2 -= c2 * np.outer(k2, k2)

            e1 = z1 - f(y1)
            e2 = z2 - f(y2)
            e3 = z3 - f(y3)
            e4 = z4 - f(y4)
            e5 = z5 - f(y5)
            e6 = z6 - f(y6)
            dw1 = -c * e1 * k
            dw2 = -c * e2 * k
            dw3 = -c * e3 * k
            dw4 = -c2 * e4 * k2
            dw5 = -c2 * e5 * k2
            dw6 = -c2 * e6 * k2

            wo1 = wo1 + dw1
            wo2 = wo2 + dw2
            wo3 = wo3 + dw3
            wo4 = wo4 + dw4
            wo5 = wo5 + dw5
            wo6 = wo6 + dw6

print("")
print("***********")
print("Testing... ")
print("***********")

I = np.zeros((nIn2Rec, simtime_len))

rand_len = np.random.randint(interval + 2, interval + 5)
rand = [''] * rand_len
for i in range(rand_len):
    rand[i] = random_elements[np.random.randint(0, len(random_elements))]
chunk = rand
m = 0
chunk1_start = []
chunk2_start = []
chunk3_start = []
for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if chunk == chunk1:
            if m == len(chunk1) - 1:
                rand_len = np.random.randint(interval + 2, interval + 5)
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
                rand_len = np.random.randint(interval + 2, interval + 5)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0
            else:
                chunk = chunk2
                m += 1

        elif chunk == chunk3:
            if m == len(chunk3) - 1:
                rand_len = np.random.randint(interval + 2, interval + 5)
                rand = [''] * rand_len
                for l in range(rand_len):
                    rand[l] = random_elements[np.random.randint(0, len(random_elements))]

                chunk = rand
                m = 0
            else:
                chunk = chunk3
                m += 1

        elif chunk == rand:
            if m == len(rand) - 1:
                dice = np.random.rand()
                if dice < 1/3:
                    chunk = chunk1
                    m = 0
                 
                    chunk1_start.append(i)
                elif dice < 2/3:
                    chunk = chunk2
                    m = 0
               
                    chunk2_start.append(i)
                else:
                    chunk = chunk3
                    m = 0
          
                    chunk3_start.append(i)
            else:
                chunk = rand
                m += 1
        # print(chunk[m])
        input_end = min(i + width * 2, simtime_len)
        I[elements_list.index(chunk[m]), i:input_end] = input_shape[0:input_end - i]
test_len = min(width * 2000, simtime_len)
z1_list = np.zeros(test_len)
z2_list = np.zeros(test_len)
z3_list = np.zeros(test_len)
z4_list = np.zeros(test_len)
z5_list = np.zeros(test_len)
z6_list = np.zeros(test_len)
r_list = np.zeros((N, test_len))
r2_list = np.zeros((N, test_len))
chunk1_avg1 = np.zeros((N, width * (len(chunk1) + interval * 4)))
chunk1_avg2 = np.zeros((N, width * (len(chunk1) + interval * 4)))
chunk2_avg1 = np.zeros((N, width * (len(chunk1) + interval * 4)))
chunk2_avg2 = np.zeros((N, width * (len(chunk1) + interval * 4)))
chunk3_avg1 = np.zeros((N, width * (len(chunk1) + interval * 4)))
chunk3_avg2 = np.zeros((N, width * (len(chunk1) + interval * 4)))

x = x01
x2 = x02
r = np.tanh(x)
r2 = np.tanh(x2)

z1 = z01
z2 = z02
z3 = z03
z4 = z04
z5 = z05
z6 = z06

for i in range(test_len):
    x = (1.0 - dt / tau) * x + np.dot(M, r) * dt / tau + np.dot(win, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf1 * z1 * dt / tau + wf2 * z2 * dt / tau + wf3 * z3 * dt / tau
    x2 = (1.0 - dt / tau) * x2 + np.dot(M2, r2) * dt / tau + np.dot(win2, I[:, i]) * dt / tau + sigma * np.sqrt(
        dt) * np.random.randn(N) + wf4 * z4 * dt / tau + wf5 * z5 * dt / tau + wf6 * z6 * dt / tau

    r = np.tanh(x)
    r2 = np.tanh(x2)

    z1 = np.dot(wo1, r[synapse_list1])
    z2 = np.dot(wo2, r[synapse_list1])
    z3 = np.dot(wo3, r[synapse_list1])
    z4 = np.dot(wo4, r2[synapse_list2])
    z5 = np.dot(wo5, r2[synapse_list2])
    z6 = np.dot(wo6, r2[synapse_list2])

    z1_list[i] = z1
    z2_list[i] = z2
    z3_list[i] = z3
    z4_list[i] = z4
    z5_list[i] = z5
    z6_list[i] = z6

    r_list[:,i] = r
    r2_list[:, i] = r2

chunk1test_start = [x for x in chunk1_start if x < test_len]
z1_avg_chunk1 = np.zeros(width * (len(chunk1) + interval * 4))
z1_avg_chunk2 = np.zeros(width * (len(chunk1) + interval * 4))
z1_avg_chunk3 = np.zeros(width * (len(chunk1) + interval * 4))

chunk2test_start = [x for x in chunk2_start if x < test_len]
z2_avg_chunk1 = np.zeros(width * (len(chunk1) + interval * 4))
z2_avg_chunk2 = np.zeros(width * (len(chunk1) + interval * 4))
z2_avg_chunk3 = np.zeros(width * (len(chunk1) + interval * 4))

chunk3test_start = [x for x in chunk3_start if x < test_len]
z3_avg_chunk1 = np.zeros(width * (len(chunk1) + interval * 4))
z3_avg_chunk2 = np.zeros(width * (len(chunk1) + interval * 4))
z3_avg_chunk3 = np.zeros(width * (len(chunk1) + interval * 4))

for j in chunk1test_start[1:-5]:
    z1_avg_chunk1 += z1_list[j - width * interval:j + width * len(chunk1) + interval * width*3]
    z2_avg_chunk1 += z2_list[j - width * interval:j + width * len(chunk1) + interval * width*3]
    z3_avg_chunk1 += z3_list[j - width * interval:j + width * len(chunk1) + interval * width*3]

    chunk1_avg1 += r_list[:, j - width * interval:j + width * len(chunk1) + interval * width*3]
    chunk1_avg2 += r2_list[:, j - width * interval:j + width * len(chunk1) + interval * width*3]

for j in chunk2test_start[1:-5]:
    z1_avg_chunk2 += z1_list[j - width * interval:j + width * len(chunk1) + interval * width*3]
    z2_avg_chunk2 += z2_list[j - width * interval:j + width * len(chunk1) + interval * width*3]
    z3_avg_chunk2 += z3_list[j - width * interval:j + width * len(chunk1) + interval * width*3]

    chunk2_avg1 += r_list[:, j - width * interval:j + width * len(chunk1) + interval * width*3]
    chunk2_avg2 += r2_list[:, j - width * interval:j + width * len(chunk1) + interval * width*3]

for j in chunk3test_start[1:-5]:
    z1_avg_chunk3 += z1_list[j - width * interval:j + width * len(chunk1) + interval * width*3]
    z2_avg_chunk3 += z2_list[j - width * interval:j + width * len(chunk1) + interval * width*3]
    z3_avg_chunk3 += z3_list[j - width * interval:j + width * len(chunk1) + interval * width*3]

    chunk3_avg1 += r_list[:, j - width * interval:j + width * len(chunk1) + interval * width*3]
    chunk3_avg2 += r2_list[:, j - width * interval:j + width * len(chunk1) + interval * width*3]

z1_avg_chunk1 /= len(chunk1test_start[1:-5])
z2_avg_chunk1 /= len(chunk1test_start[1:-5])
z3_avg_chunk1 /= len(chunk1test_start[1:-5])
z1_avg_chunk2 /= len(chunk2test_start[1:-5])
z2_avg_chunk2 /= len(chunk2test_start[1:-5])
z3_avg_chunk2 /= len(chunk2test_start[1:-5])
z1_avg_chunk3 /= len(chunk3test_start[1:-5])
z2_avg_chunk3 /= len(chunk3test_start[1:-5])
z3_avg_chunk3 /= len(chunk3test_start[1:-5])

chunk1_avg1 /= len(chunk1test_start[1:-5])
chunk1_avg2 /= len(chunk1test_start[1:-5])
chunk2_avg1 /= len(chunk2test_start[1:-5])
chunk2_avg2 /= len(chunk2test_start[1:-5])
chunk3_avg1 /= len(chunk3test_start[1:-5])
chunk3_avg2 /= len(chunk3test_start[1:-5])

avg_chunk_range = np.zeros(width * (len(chunk1) + interval * 4)) - 1
avg_chunk_range[interval * width:width * len(chunk1) + interval * width] = 1

fig = plt.figure(figsize=(7, 2))
ax = plt.subplot(1, 1, 1)
pl.plot(z1_list, lw=1.5, c='orangered')
pl.plot(z2_list, lw=1.5, c='limegreen')
pl.plot(z3_list, lw=1.5, c='dodgerblue')

for i in (chunk1test_start):
    pl.axvspan(i, i + width * 4, facecolor='limegreen', alpha=0.3,linewidth=0)
for i in (chunk2test_start):
    pl.axvspan(i, i + width * 4, facecolor='dodgerblue', alpha=0.3,linewidth=0)
for i in (chunk3test_start):
    pl.axvspan(i, i + width * 4, facecolor='orangered', alpha=0.3,linewidth=0)

pl.ylim([-0.3,1.0])
plot_start = chunk1test_start[4] - width * interval
pl.xlim(plot_start, plot_start + 3000)


plt.xlabel("Time [ms]", fontsize=10)
plt.ylabel("Activity", fontsize=10)

plt.xticks([plot_start,plot_start+1000,plot_start+2000,plot_start+3000],['0','1000','2000','3000'],fontsize=10)
fig.subplots_adjust(bottom=0.25, left=0.15)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('readouts.pdf', fmt='pdf', dpi=350)
