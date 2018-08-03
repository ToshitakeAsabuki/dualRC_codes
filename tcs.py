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

sigma = 0.1

input_gain = 2
interval = 3
symbol_list = ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m','n','o']
nIn2Rec = len(symbol_list)
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

symbol='a'

for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if symbol=='a':
            symbol=np.random.choice(['b','c','d','e'],1)
        elif symbol=='b':
            symbol=np.random.choice(['c','d','a','e'],1)
        elif symbol=='c':
            symbol=np.random.choice(['b','a','e','f'],1)
        elif symbol=='d':
            symbol=np.random.choice(['b','a','e','k'],1)
        elif symbol=='e':
            symbol=np.random.choice(['b','a','c','d'],1)
        elif symbol=='f':
            symbol=np.random.choice(['c','g','h','i'],1)
        elif symbol=='g':
            symbol=np.random.choice(['f','j','i','h'],1)
        elif symbol=='h':
            symbol=np.random.choice(['g','f','i','j'],1)
        elif symbol=='i':
            symbol=np.random.choice(['f','g','h','j'],1)
        elif symbol=='j':
            symbol=np.random.choice(['g','h','i','l'],1)
        elif symbol=='k':
            symbol=np.random.choice(['d','m','n','o'],1)
        elif symbol=='l':
            symbol=np.random.choice(['j','m','n','o'],1)
        elif symbol=='m':
            symbol=np.random.choice(['l','k','n','o'],1)
        elif symbol=='n':
            symbol=np.random.choice(['m','l','k','o'],1)
        elif symbol=='o':
            symbol=np.random.choice(['k','l','m','n'],1)

        input_end = min(i + width * 2, simtime_len)
        I[symbol_list.index(symbol), i:input_end] = input_shape[0:input_end - i]

print("")
print("***********")
print("Learning... ")
print("***********")
z1_list_learning = np.zeros(simtime_len)
z2_list_learning = np.zeros(simtime_len)
z3_list_learning = np.zeros(simtime_len)
z4_list_learning = np.zeros(simtime_len)
z5_list_learning = np.zeros(simtime_len)
z6_list_learning = np.zeros(simtime_len)
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

    z1_list_learning[i] = z1
    z2_list_learning[i] = z2
    z3_list_learning[i] = z3
    z4_list_learning[i] = z4
    z5_list_learning[i] = z5
    z6_list_learning[i] = z6


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

    y1 = (z4 - np.mean(z4_list)) / np.std(z4_list) - lateral * (
    (z5 - np.mean(z5_list)) / np.std(z5_list) + (z6 - np.mean(z6_list)) / np.std(z6_list))
    y2 =(z5 - np.mean(z5_list)) / np.std(z5_list) - lateral * (
    (z4 - np.mean(z4_list)) / np.std(z4_list) + (z6 - np.mean(z6_list)) / np.std(z6_list))
    y3 =(z6 - np.mean(z6_list)) / np.std(z6_list) - lateral * (
    (z5 - np.mean(z5_list)) / np.std(z5_list) + (z4 - np.mean(z4_list)) / np.std(z4_list))
    y4 = (z1 - np.mean(z1_list)) / np.std(z1_list) - lateral * (
    (z2 - np.mean(z2_list)) / np.std(z2_list) + (z3 - np.mean(z3_list)) / np.std(z3_list))
    y5 = (z2 - np.mean(z2_list)) / np.std(z2_list) - lateral * (
    (z1 - np.mean(z1_list)) / np.std(z1_list) + (z3 - np.mean(z3_list)) / np.std(z3_list))
    y6 = (z3 - np.mean(z3_list)) / np.std(z3_list) - lateral * (
    (z1 - np.mean(z1_list)) / np.std(z1_list) + (z2 - np.mean(z2_list)) / np.std(z2_list))
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
comm1_start=[0]
comm2_start=[]
comm3_start=[]

symbol='a'
for i in range(simtime_len):
    if (i % width == 0 and i > 0):
        if symbol=='a':
            symbol=np.random.choice(['b','c','d','e'],1)
        elif symbol=='b':
            symbol=np.random.choice(['c','d','a','e'],1)
        elif symbol=='c':
            symbol=np.random.choice(['b','a','e','f'],1)
        elif symbol=='d':
            symbol=np.random.choice(['b','a','e','k'],1)
        elif symbol=='e':
            symbol=np.random.choice(['b','a','c','d'],1)
        elif symbol=='f':
            symbol=np.random.choice(['c','g','h','i'],1)
        elif symbol=='g':
            symbol=np.random.choice(['f','j','i','h'],1)
        elif symbol=='h':
            symbol=np.random.choice(['g','f','i','j'],1)
        elif symbol=='i':
            symbol=np.random.choice(['f','g','h','j'],1)
        elif symbol=='j':
            symbol=np.random.choice(['g','h','i','l'],1)
        elif symbol=='k':
            symbol=np.random.choice(['d','m','n','o'],1)
        elif symbol=='l':
            symbol=np.random.choice(['j','m','n','o'],1)
        elif symbol=='m':
            symbol=np.random.choice(['l','k','n','o'],1)
        elif symbol=='n':
            symbol=np.random.choice(['m','l','k','o'],1)
        elif symbol=='o':
            symbol=np.random.choice(['k','l','m','n'],1)

        if symbol in ['a', 'b', 'c', 'd','e']:
            comm1_start.append(i)
        elif symbol in ['f', 'g', 'h', 'i','j']:
            comm2_start.append(i)
        else:
            comm3_start.append(i)

        input_end = min(i + width * 2, simtime_len)
        I[symbol_list.index(symbol), i:input_end] = input_shape[0:input_end - i]
test_len = min(width * 2000, simtime_len)
z1_list = np.zeros(test_len)
z2_list = np.zeros(test_len)
z3_list = np.zeros(test_len)
z4_list = np.zeros(test_len)
z5_list = np.zeros(test_len)
z6_list = np.zeros(test_len)
r_list = np.zeros((N, test_len))
r2_list = np.zeros((N, test_len))


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


fig = plt.figure(figsize=(7, 2))
ax = plt.subplot(1, 1, 1)
pl.plot(z1_list, lw=1.5, c='orangered')
pl.plot(z2_list, lw=1.5, c='limegreen')
pl.plot(z3_list, lw=1.5, c='dodgerblue')

for i in (comm1_start):
    pl.axvspan(i, i + width , facecolor='orangered', alpha=0.3,linewidth=0)
for i in (comm2_start):
    pl.axvspan(i, i + width , facecolor='dodgerblue', alpha=0.3,linewidth=0)
for i in (comm3_start):
    pl.axvspan(i, i + width , facecolor='limegreen', alpha=0.3,linewidth=0)


pl.ylim(min(np.min(z1_list[0:test_len]), np.min(z2_list[0:test_len]), np.min(z3_list[0:test_len])) - 0.1,
            max(np.max(z1_list[0:test_len]), np.max(z2_list[0:test_len]), np.max(z3_list[0:test_len])) + 0.1)
plot_start = 0
pl.xlim(plot_start, plot_start + 5000)


plt.xlabel("Time [ms]", fontsize=15)
plt.ylabel("Activity", fontsize=15)

fig.subplots_adjust(bottom=0.25, left=0.15)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('readouts.pdf', fmt='pdf', dpi=350)


