import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dateutil.parser
import time


def to_seconds(date):
    return time.mktime(date.timetuple())


def parse_log_timestamp(log_time):
    return to_seconds(dateutil.parser.parse(log_time))

colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
markerList = ['v', 'o', '.', 's', '1', "*", "+"]
f = open("progress_2.csv")

cut = 68
x = []
x_sum = []
y = []
y_sum = []
start_time = None
prev_time = 0
prev_count = 0
i = 0
for line in f.readlines():
    i += 1
    a = line.split()
    t = parse_log_timestamp(a[0])
    count = int(a[15])
    if i == cut:
        start_time = t
    if start_time is None:
        continue
    t = t - start_time
    y_sum.append(count)
    x_sum.append(t)
    if i % 50 != 0:
        continue
    x.append(t)
    y.append((count - prev_count) / (t - prev_time))
    prev_count = count
    prev_time = t

f = open("progress_def_1.csv")
def_x = []
def_x_sum = []
def_y = []
def_y_sum = []
start_time = None
prev_time = 0
prev_count = 0
i = 0
for line in f.readlines():
    i += 1
    a = line.split()
    t = parse_log_timestamp(a[0])
    count = int(a[15])
    if i == cut:
        start_time = t
    if start_time is None:
        continue
    t = t - start_time
    def_y_sum.append(count)
    def_x_sum.append(t)
    if i % 50 != 0:
        continue
    def_x.append(t)
    def_y.append((count - prev_count) / (t - prev_time))
    prev_time = t
    prev_count = count

f = open("with_random_tx_700_50.csv")
combined_x_sum = []
combined_y_sum = []
combined_x = []
combined_y = []
start_time = None
prev_time = 0
prev_count = 0
i = 0
for line in f.readlines():
    i += 1
    a = line.split()
    t = parse_log_timestamp(a[0])
    count = int(a[15])
    if i == cut:
        start_time = t
    if start_time is None:
        continue
    t = t - start_time
    combined_y_sum.append(count)
    combined_x_sum.append(t)
    if i % 50 != 0:
        continue
    combined_x.append(t)
    combined_y.append((count - prev_count) / (t - prev_time))
    prev_time = t
    prev_count = count

matplotlib.rcParams.update({'font.size': 12})

plt.figure(figsize=(6, 3))
output = {}
ax = plt.subplot(111)
ax.plot(x[2:], y[2:], label="ETH (k=5)")
ax.plot(def_x[2:], def_y[2:], label="ETH (k=1)")
ax.plot(combined_x[2:], combined_y[2:], label="ETH + Payment (k=5)")

ax.set_xlabel("Timestamp (s)")
ax.set_ylabel("Tx Per Second")
plt.legend()
plt.tight_layout()
plt.savefig("end_to_end_tps.pdf")

plt.figure(figsize=(6, 3))
ax = plt.subplot(111)
ax.plot(x_sum[cut:], y_sum[cut:], label="ETH (k=5)")
print({"ETH (k=5)": [x_sum[2:], y_sum[2:]]})

ax.plot(def_x_sum[cut:], def_y_sum[cut:], label="ETH (k=1)")
print({"ETH (k=1)": [def_x_sum[2:], def_y_sum[2:]]})

ax.plot(combined_x_sum[cut:], combined_y_sum[
        cut:], label="ETH + Payment (k=5)")
print({"ETH + Payment (k=5)": [combined_x_sum[2:], combined_y_sum[2:]]})

ax.set_xlabel("Timestamp (s)")
ax.set_ylabel("Total Tx Count")
plt.legend(prop={'size': 11})
plt.tight_layout(pad=0)
# plt.show()
plt.savefig("end_to_end_cumulative.pdf")
exit()

heavy_block_confirm_time = {
    0.04: [11.00, 68.00, 19.00, 25.00, 22.00, ],
    0.06: [11.00, 11.00, 33.00, ],
    0.08: [156.00, 62.00, 16.00, ],
    0.10: [10.00, 13.00, 90.00, ],
    0.12: [88.00, 15.00, 66.00, 18.00, ],
    0.14: [29.00, 14.00, 111.00, 10.00, 11.00, ],
    0.16: [56.00, 14.00, 48.00, ],
    0.18: [37.00, 262.00, 33.00, 12.00, ],
    0.20: [332.00, 16.00, 51.00, 37.00, ],
}

plt.figure(figsize=(10, 5))
ax = plt.subplot(111)
name = 'Conflux'
x = []
y = []
y_box = []
color = colorList[0]
marker = markerList[0]
x = []
y = []
y_box = []
for n in sorted(heavy_block_confirm_time.keys()):
    time_array = sorted(heavy_block_confirm_time[n])
    avg = sum(time_array) / len(time_array)
    median = time_array[len(time_array) / 2]
    p25 = time_array[len(time_array) / 4]
    p75 = time_array[len(time_array) / 4 * 3]
    min_time = time_array[0]
    max_time = time_array[-1]
    x.append(n)
    y.append(avg)
    y_box.append([min_time, p25, avg, p75, max_time])
w = 0.01
width = lambda p, w: 10**(np.log10(p) + w / 2.) - \
    10**(np.log10(p) - w / 2.)
print(x, y, y_box)
bp = ax.boxplot(y_box, positions=x, medianprops=dict(
    linestyle=None, linewidth=0), whis='range', sym=(''), widths=[0.003] * len(x))
for box in bp['boxes']:
    # change outline color
    box.set(color=color, linewidth=1)
for whisker in bp['whiskers']:
    whisker.set(color=color, linewidth=1)
for cap in bp['caps']:
    cap.set(color=color, linewidth=1)
ax.plot(x, y, label=name, marker=marker, color=color, markersize=3)
# ax.set_xticklabels(["10s", "20s", "40s", "80s"])
plt.legend()
ax.set_xlim(0.04, 0.21)
# ax.set_ylim(0, 1000)
ax.set_xlabel("Adversary Power")
ax.set_ylabel("Confirmation Time (s)")
# ax.set_xscale('log')
# ax.set_xticks(nums)
# ax.set_xticklabels(['2.5k', '5k', '10k', '20k'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.tight_layout()
plt.show()
plt.savefig("balance_attack.pdf")


exit()


f = open("data", 'r')
latency = {}
throughput = {}
chain_ratio = {}
D = {}
for line in f.readlines():
    a = line.split("\t")
    name = a[0]
    block_size = a[1]
    interval = a[2]
    latency_t = a[4:9]
    D_t = a[3]
    chain_ratio_t = a[9]
    # input unit is MB/s
    throughput_t = a[10]
    if name not in latency:
        latency[name] = {}
        throughput[name] = {}
        chain_ratio[name] = {}
        D[name] = {}
    if block_size not in latency[name]:
        latency[name][block_size] = {}
        throughput[name][block_size] = {}
        chain_ratio[name][block_size] = {}
        D[name][block_size] = {}
    latency[name][block_size][interval] = latency_t
    chain_ratio[name][block_size][interval] = chain_ratio_t
    throughput[name][block_size][interval] = throughput_t
    D[name][block_size][interval] = D_t
f.close()

names = ['Conflux', 'GHOST', 'Bitcoin']
colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
markerList = ['v', 'o', '.', 's', '1', "*", "+"]
blockSizes = [1, 2, 4, 8]
intervals = [5, 10, 20, 40, 80]
colors = {}
markers = {}
for i in range(len(names)):
    colors[names[i]] = colorList[i]
    markers[names[i]] = markerList[i]

for name in names:
    for block_size in throughput[name]:
        for interval in throughput[name][block_size]:
            if name != 'Conflux':
                throughput[name][block_size][interval] = float(chain_ratio[name][block_size][
                    interval]) * int(block_size) * 3600 / float(interval)
            else:
                throughput[name][block_size][interval] = int(
                    block_size) * 3600 / float(interval)

plt.figure(figsize=(5, 5))
ax = plt.subplot(211)
ax.set_xlim(1 / 1.5, 8 * 1.5)
# ax.set_ylim(0,1.05)
ax.set_xlabel("Block Size (with 20s block generation interval)")
ax.set_ylabel("Throguhput (MB/h)")
ax.set_xscale('log')
ax.set_xticks(blockSizes)
ax.set_xticklabels([str(i) + "MB" for i in blockSizes])
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(['0','20','40','60','80','100'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
# ax.xaxis.set_major_locator(plt.FixedLocator(blockSizes))
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%dMB'))
# plt.plot(blockSizes, [1 for _ in blockSizes], label='Conflux',
# color=colors['Conflux'], marker=markers['Conflux'])
for n in names:
    x = []
    y = []
    for bs in blockSizes:
        x.append(bs)
        # convert to MB/h
        y.append(float(throughput[n][str(bs)]['20']))
    print(x, y)
    plt.plot(x, y, label=n, color=colors[n], marker=markers[n])
plt.legend()
plt.tight_layout()
# plt.savefig("chain_ratio_bs.pdf")
# plt.figure(figsize=(5,2.5))
ax = plt.subplot(212)
ax.set_xlim(5 / 1.5, 80 * 1.5)
# ax.set_ylim(0,1.05)
ax.set_xlabel("Generation Interval (with 4MB block size)")
ax.set_ylabel("Throguhput (MB/h)")
ax.set_xscale('log')
ax.set_xticks(intervals)
ax.set_xticklabels([str(i) + "s" for i in intervals])
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(['0','20','40','60','80','100'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
# plt.plot(intervals, [1 for _ in intervals], label='Conflux',
# color=colors['Conflux'], marker=markers['Conflux'])
for n in names:
    x = []
    y = []
    for i in intervals:
        # if i == 5 and n == 'Bitcoin':
        # 	continue
        x.append(i)
        # convert to MB/h
        y.append(float(throughput[n]['4'][str(i)]))
    print(x, y)
    plt.plot(x, y, label=n, color=colors[n], marker=markers[n])
plt.legend()
plt.tight_layout()
plt.savefig("throughput.pdf")

plt.figure(figsize=(5, 5))
ax = plt.subplot(211)
ax.set_xlim(1 / 1.5, 8 * 1.5)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Block Size (with 20s block generation interval)")
ax.set_ylabel("Block Utilization (%)")
ax.set_xscale('log')
ax.set_xticks(blockSizes)
ax.set_xticklabels([str(i) + "MB" for i in blockSizes])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
# ax.xaxis.set_major_locator(plt.FixedLocator(blockSizes))
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%dMB'))
plt.plot(blockSizes, [1 for _ in blockSizes], label='Conflux',
         color=colors['Conflux'], marker=markers['Conflux'])
for n in names:
    x = []
    y = []
    for bs in blockSizes:
        x.append(bs)
        # convert to MB/h
        y.append(float(chain_ratio[n][str(bs)]['20']))
    print(x, y)
    if n == 'Conflux':
        continue
        # plt.plot(x, y, label='Conflux Pivot Chain', color=colorList[3],
        # marker=markerList[3])
    else:
        plt.plot(x, y, label=n, color=colors[n], marker=markers[n])
plt.legend(loc=3)
plt.tight_layout()
# plt.savefig("chain_ratio_bs.pdf")
# plt.figure(figsize=(5,2.5))
ax = plt.subplot(212)
ax.set_xlim(5 / 1.5, 80 * 1.5)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Generation Interval (with 4MB block size)")
ax.set_ylabel("Block Utilization (%)")
ax.set_xscale('log')
ax.set_xticks(intervals)
ax.set_xticklabels([str(i) + "s" for i in intervals])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.plot(intervals, [1 for _ in intervals], label='Conflux',
         color=colors['Conflux'], marker=markers['Conflux'])
for n in names:
    x = []
    y = []
    for i in intervals:
        # if i == 5 and n == 'Bitcoin':
        # 	continue
        x.append(i)
        # convert to MB/h
        y.append(float(chain_ratio[n]['4'][str(i)]))
    print(x, y)
    if n == 'Conflux':
        continue
        # plt.plot(x, y, label='Conflux Pivot Chain', color=colorList[3],
        # marker=markerList[3])
    else:
        plt.plot(x, y, label=n, color=colors[n], marker=markers[n])
plt.legend(loc=4)
plt.tight_layout()
plt.savefig("chain_ratio.pdf")

plt.figure(figsize=(5, 5))
ax = plt.subplot(211)
for n in names:
    # for n in ['Conflux']:
    x = []
    y = []
    y_box = []
    for bs in blockSizes:
        if n == 'Bitcoin' and bs in [4, 8]:
            continue
        x.append(bs)
        # convert to MB/h
        y.append(float(latency[n][str(bs)]['20'][2]))
        y_box.append([float(_) for _ in latency[n][str(bs)]['20']])
    print(x, y)
    w = 0.04
    width = lambda p, w: 10**(np.log10(p) + w / 2.) - \
        10**(np.log10(p) - w / 2.)
    bp = ax.boxplot(y_box, positions=x, medianprops=dict(
        linestyle=None, linewidth=0), whis='range', sym=(''), widths=width(x, w))
    for box in bp['boxes']:
        # change outline color
        box.set(color=colors[n], linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color=colors[n], linewidth=1)
    for cap in bp['caps']:
        cap.set(color=colors[n], linewidth=1)
    ax.plot(x, y, label=n, color=colors[n], marker=markers[n])
    # ax.set_xticklabels(["1MB", "2MB", "4MB", "8MB"])
ax.set_xlim(1 / 1.5, 8 * 1.5)
ax.set_ylim(0, 3500)
ax.set_xlabel("Block Size (with 20s block generation interval)")
ax.set_ylabel("Confirmation Time (s)")
ax.set_xscale('log')
ax.set_xticks(blockSizes)
ax.set_xticklabels([str(i) + "MB" for i in blockSizes])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.legend()
plt.tight_layout()
# plt.savefig("confirmation_time_bs.pdf")
# plt.show()
# plt.figure(figsize=(5,2.5))
ax = plt.subplot(212)
for n in names:
    x = []
    y = []
    y_box = []
    for i in intervals:
        if i != 80 and n == 'Bitcoin':
            continue
        x.append(i)
        # convert to MB/h
        y.append(float(latency[n]['4'][str(i)][2]))
        y_box.append([float(_) for _ in latency[n]['4'][str(i)]])
    print(x, y)
    w = 0.04
    width = lambda p, w: 10**(np.log10(p) + w / 2.) - \
        10**(np.log10(p) - w / 2.)
    bp = ax.boxplot(y_box, positions=x, medianprops=dict(
        linestyle=None, linewidth=0), whis='range', sym=(''), widths=width(x, w))
    for box in bp['boxes']:
        # change outline color
        box.set(color=colors[n], linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color=colors[n], linewidth=1)
    for cap in bp['caps']:
        cap.set(color=colors[n], linewidth=1)
    ax.plot(x, y, label=n, color=colors[n], marker=markers[n])
    # ax.set_xticklabels(["10s", "20s", "40s", "80s"])
plt.legend()
ax.set_xlim(5 / 1.5, 80 * 1.5)
ax.set_ylim(0, 4500)
ax.set_xlabel("Generation Interval (with 4MB block size)")
ax.set_ylabel("Confirmation Time (s)")
ax.set_xscale('log')
ax.set_xticks(intervals)
ax.set_xticklabels([str(i) + "s" for i in intervals])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.tight_layout()
# plt.show()
plt.savefig("confirmation_time.pdf")


plt.figure(figsize=(5, 5))
ax = plt.subplot(211)
ax.set_xlim(1 / 1.5, 8 * 1.5)
ax.set_ylim(0, 350)
ax.set_xlabel("Block Size (with 20s block generation interval)")
ax.set_ylabel("99th Percentile Latency")
ax.set_xscale('log')
ax.set_xticks(blockSizes)
ax.set_xticklabels([str(i) + "MB" for i in blockSizes])
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(['0','20','40','60','80','100'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
# ax.xaxis.set_major_locator(plt.FixedLocator(blockSizes))
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%dMB'))
# plt.plot(blockSizes, [1 for _ in blockSizes], label='Conflux',
# color=colors['Conflux'], marker=markers['Conflux'])
for n in names:
    if n == 'GHOST':
        continue
    x = []
    y = []
    for bs in blockSizes:
        x.append(bs)
        # convert to MB/h
        y.append(float(D[n][str(bs)]['20']))
    print(x, y)
    # if n == 'Conflux':
    # continue
    # plt.plot(x, y, label='Conflux Pivot Chain', color=colorList[3], marker=markerList[3])
    # else:
    plt.plot(x, y, label=n, color=colors[n], marker=markers[n])
plt.legend()
plt.tight_layout()
# plt.savefig("chain_ratio_bs.pdf")
# plt.figure(figsize=(5,2.5))
ax = plt.subplot(212)
ax.set_xlim(5 / 1.5, 80 * 1.5)
ax.set_ylim(0, 350)
ax.set_xlabel("Generation Interval (with 4MB block size)")
ax.set_ylabel("99th Percentile Latency (s)")
ax.set_xscale('log')
ax.set_xticks(intervals)
ax.set_xticklabels([str(i) + "s" for i in intervals])
# ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(['0','20','40','60','80','100'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
# plt.plot(intervals, [1 for _ in intervals], label='Conflux',
# color=colors['Conflux'], marker=markers['Conflux'])
for n in names:
    if n == 'GHOST':
        continue
    x = []
    y = []
    for i in intervals:
        # if i == 5 and n == 'Bitcoin':
        # 	continue
        x.append(i)
        # convert to MB/h
        y.append(float(D[n]['4'][str(i)]))
    print(x, y)
    # if n == 'Conflux':
    # continue
    # plt.plot(x, y, label='Conflux Pivot Chain', color=colorList[3], marker=markerList[3])
    # else:
    plt.plot(x, y, label=n, color=colors[n], marker=markers[n])
plt.legend()
plt.tight_layout()
plt.savefig("D_bs_interval.pdf")


f = open("data_num", 'r')
latency = {}
chain_ratio = {}
nums = [2500, 5000, 10000, 20000]
for line in f.readlines():
    a = line.split("\t")
    n = a[0]
    latency_t = a[2:7]
    D_t = a[1]
    chain_ratio_t = a[7]
    if n not in latency:
        latency[n] = {}
        chain_ratio[n] = {}
        D[n] = {}
    latency[n] = latency_t
    chain_ratio[n] = chain_ratio_t
    D[n] = D_t
f.close()
for i in range(len(nums)):
    colors[nums[i]] = colorList[i]
    markers[nums[i]] = markerList[i]

plt.figure(figsize=(5, 2.5))
# ax = plt.subplot(211)
# ax.set_xlim(2500/1.5,20000*1.5)
# ax.set_ylim(0,1)
# # ax.set_xlabel("Block Size")
# ax.set_ylabel("Chain Ratio")
# ax.set_xscale('log')
# ax.set_xticks(nums)
# ax.set_xticklabels(['2.5k','5k','10k','20k'])
# ax.xaxis.set_tick_params(which='minor', size=0)
# ax.xaxis.set_tick_params(which='minor', width=0)
# x = []
# y = []
# for n in nums:
# 	x.append(n)
# 	# convert to MB/h
# 	y.append(float(chain_ratio[str(n)]))
# print(x, y)
# plt.plot(x, y, label='Conflux',marker=markerList[0],color=colorList[0])
# plt.legend()
ax = plt.subplot(111)
x = []
y = []
y_box = []
for n in nums:
    x.append(n)
    # convert to MB/h
    y.append(float(latency[str(n)][2]))
    y_box.append([float(_) for _ in latency[str(n)]])
print(x, y)
w = 0.04
width = lambda p, w: 10**(np.log10(p) + w / 2.) - 10**(np.log10(p) - w / 2.)
bp = ax.boxplot(y_box, positions=x, medianprops=dict(
    linestyle=None, linewidth=0), whis='range', sym=(''), widths=width(x, w))
for box in bp['boxes']:
    # change outline color
    box.set(color=colorList[0], linewidth=1)
for whisker in bp['whiskers']:
    whisker.set(color=colorList[0], linewidth=1)
for cap in bp['caps']:
    cap.set(color=colorList[0], linewidth=1)
ax.plot(x, y, label='Conflux', marker=markerList[0], color=colorList[0])
# ax.set_xticklabels(["10s", "20s", "40s", "80s"])
plt.legend()
ax.set_xlim(2500 / 1.5, 20000 * 1.5)
ax.set_ylim(0, 1000)
ax.set_xlabel("Number of Users")
ax.set_ylabel("Confirmation Time (s)")
ax.set_xscale('log')
ax.set_xticks(nums)
ax.set_xticklabels(['2.5k', '5k', '10k', '20k'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.tight_layout()
# plt.show()
plt.savefig("plot_num.pdf")

f = open("data_4M_10s", 'r')
q = [0.1, 0.2, 0.3, 0.4]
r = [0.01, 0.001, 0.0001, 0.00001]
wt = [_ for _ in range(250, 2000, 250)]
r.reverse()
latency_r = {_: {} for _ in names}
latency_q = {_: {} for _ in names}
risk_w = {_: {} for _ in names}
name = 'Conflux'
for line in f.readlines():
    if "Conflux" in line:
        name = 'Conflux'
        continue
    elif "GHOST" in line:
        name = 'GHOST'
        continue
    elif 'Bitcoin' in line:
        name = 'Bitcoin'
        break
        continue
    a = line.split('\t')
    if float(a[0]) in q:
        latency_q[name][float(a[0])] = a[1:6]
    elif float(a[0]) in r:
        latency_r[name][float(a[0])] = a[1:6]
    elif int(a[0]) in wt:
        risk_w[name][int(a[0])] = a[1:6]
print(latency_r)
print(latency_q)
print(risk_w)


def my_float(f):
    if float(f) < 1e-15:
        return 1e-15
    return float(f)
plt.figure(figsize=(5, 5))
ax = plt.subplot(211)
for name in names:
    x = []
    y = []
    y_box = []
    if name == 'Bitcoin':
        continue
    for n in q:
        if name == 'Bitcoin' and n >= 0.3:
            continue
        x.append(n)
        # convert to MB/h
        y.append(float(latency_q[name][n][2]))
        y_box.append([float(_) for _ in latency_q[name][n]])
    print(x, y)
    w = 0.01
    width = lambda p, w: 10**(np.log10(p) + w / 2.) - \
        10**(np.log10(p) - w / 2.)
    bp = ax.boxplot(y_box, positions=x, medianprops=dict(
        linestyle=None, linewidth=0), whis='range', sym=(''), widths=[0.01] * len(x))
    for box in bp['boxes']:
        # change outline color
        box.set(color=colors[name], linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color=colors[name], linewidth=1)
    for cap in bp['caps']:
        cap.set(color=colors[name], linewidth=1)
    ax.plot(x, y, label=name, marker=markers[
            name], color=colors[name], markersize=3)
# ax.set_xticklabels(["10s", "20s", "40s", "80s"])
plt.legend()
ax.set_xlim(0.05, 0.45)
ax.set_ylim(0, 3000)
ax.set_xlabel("Attacker Power (with 0.01% risk tolerance)")
ax.set_ylabel("Confirmation Time (s)")
# ax.set_xscale('log')
ax.set_xticks(q)
# ax.set_xticklabels(['2.5k','5k','10k','20k'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.tight_layout()
# plt.show()
# plt.savefig("latency_q.pdf")

# plt.figure(figsize=(5,2.5))
ax = plt.subplot(212)
for name in names:
    x = []
    y = []
    y_box = []
    if name == 'Bitcoin':
        continue
    for n in wt:
        x.append(n)
        # convert to MB/h
        y.append(my_float(risk_w[name][n][3]))
        y_box.append([my_float(_) for _ in risk_w[name][n]])
    print(x, y)
    # w = 0.08
    # width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
    # bp = ax.boxplot(y_box, positions=x, medianprops=dict(linestyle=None,linewidth=0),whis='range',sym=(''),widths=[30] * len(x))
    # for box in bp['boxes']:
    # 	# change outline color
    # 	box.set(color=colors[name], linewidth=1)
    # for whisker in bp['whiskers']:
    # 	whisker.set(color=colors[name], linewidth=1)
    # for cap in bp['caps']:
    # 	cap.set(color=colors[name], linewidth=1)
    ax.plot(x, y, label=name, marker=markers[
            name], color=colors[name], markersize=3)
name = 'Bitcoin'
ax.plot(wt, [1 for _ in wt], label=name, marker=markers[
        name], color=colors[name], markersize=3)
# ax.set_xticklabels(["10s", "20s", "40s", "80s"])
plt.legend()
ax.set_xlim(0, 2000)
ax.set_ylim(1e-14, 5)
ax.set_xlabel("Waiting Time (with attacker having 20% power)")
ax.set_ylabel("Risk")
ax.set_yscale('log')
ax.set_yticks([1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15])
ax.set_xticks(wt)
ax.set_xticklabels([str(i) + 's' for i in wt])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.tight_layout()
plt.savefig("latency_r_q.pdf")
# # plt.figure(figsize=(5,2.5))
# ax = plt.subplot(212)
# for name in names:
# 	x = []
# 	y = []
# 	y_box = []
# 	for n in r:
# 		x.append(n)
# 		# convert to MB/h
# 		y.append(float(latency_r[name][n][2]))
# 		y_box.append([float(_) for _ in latency_r[name][n]])
# 	print(x, y)
# 	w = 0.08
# 	width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
# 	bp = ax.boxplot(y_box, positions=x, medianprops=dict(linestyle=None,linewidth=0),whis='range',sym=(''),widths=width(x, w))
# 	for box in bp['boxes']:
# 		# change outline color
# 		box.set(color=colors[name], linewidth=1)
# 	for whisker in bp['whiskers']:
# 		whisker.set(color=colors[name], linewidth=1)
# 	for cap in bp['caps']:
# 		cap.set(color=colors[name], linewidth=1)
# 	ax.plot(x, y,label=name,marker=markers[name],color=colors[name],markersize=3)
# # ax.set_xticklabels(["10s", "20s", "40s", "80s"])
# plt.legend()
# ax.set_xlim(0.000005, 0.05)
# ax.set_ylim(0,2000)
# ax.set_xlabel("Tolerated Risk")
# ax.set_ylabel("Confirmation Time (s)")
# ax.set_xscale('log')
# ax.set_xticks(r)
# ax.set_xticklabels(['0.001%','0.01%','0.1%','1%'])
# ax.xaxis.set_tick_params(which='minor', size=0)
# ax.xaxis.set_tick_params(which='minor', width=0)
# plt.tight_layout()
# plt.show()
# plt.savefig("latency_r.pdf")

plt.figure(figsize=(5, 2.5))
ax = plt.subplot(111)
x = []
y = []
y_box = []
for n in nums:
    x.append(n)
    # convert to MB/h
    y.append(float(D[str(n)]))
    # y_box.append([float(_) for _ in D[str(n)]])
print(x, y)
# w = 0.04
# width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
# bp = ax.boxplot(y_box, positions=x, medianprops=dict(linestyle=None,linewidth=0),whis='range',sym=(''),widths=width(x, w))
# for box in bp['boxes']:
# 	# change outline color
# 	box.set(color=colorList[0], linewidth=1)
# for whisker in bp['whiskers']:
# 	whisker.set(color=colorList[0], linewidth=1)
# for cap in bp['caps']:
# 	cap.set(color=colorList[0], linewidth=1)
ax.plot(x, y, label='Conflux', marker=markerList[0], color=colorList[0])
# ax.set_xticklabels(["10s", "20s", "40s", "80s"])
plt.legend()
ax.set_xlim(2500 / 1.5, 20000 * 1.5)
ax.set_ylim(80, 120)
ax.set_xlabel("Number of Users")
ax.set_ylabel("99th Percentile Latency (s)")
ax.set_xscale('log')
ax.set_xticks(nums)
ax.set_xticklabels(['2.5k', '5k', '10k', '20k'])
ax.xaxis.set_tick_params(which='minor', size=0)
ax.xaxis.set_tick_params(which='minor', width=0)
plt.tight_layout()
# plt.show()
plt.savefig("D_num.pdf")
