import pickle
from sys import argv
from queue import Queue
import os
import math


def accept(t, lambda_n, sib_tree_size, max_n, r):
    if t < 0:
        return False
    n_m = max_n - sib_tree_size
    q = 0.1
    s = 1.0
    for k in range(n_m + 1):
        a = math.exp(-1 * q * lambda_n * t)
        b = 1.0
        for j in range(1, k + 1):
            b *= q * lambda_n * t
            b /= j
        # b = math.pow(q * lambda_n * t, k)
        # c = math.factorial(k)
        d = max(1 - math.pow(0.11, n_m - k + 1) * 14, 0)
        s -= b * a * d
        if s < r:
            return True
    return False

# def accept(t, lambda_n, sib_tree_size, max_n, r):
#     if t < 0:
#         return False
#     q = 0.1
#     n_m = max_n - sib_tree_size
#     risk = 0
#     for k in range(n_m + 1):
#         d = min(16 * math.pow(0.18, n_m - k + 1), 1)
#         a = math.exp(-1 * q * lambda_n * t)
#         b = 1.0
#         for j in range(1, k + 1):
#             b *= q * lambda_n * t
#             b /= j
#         risk += d * b * a
#     print("risk", n_m, risk)
#     if risk < r:
#         return True
#     else:
#         return False


def compute_latency(aim_blocks, parents, refs, final_block, g_time, r_time, difficulties, lambda_n=20, risk=0.0001):
    aim_blocks = g_time.keys()
    r = risk
    final_block = None

    chain = []
    index = final_block
    while index in parents:
        chain.append(index)
        index = parents[index]
    chain.reverse()
    childs = {}
    des = {}
    for i in parents:
        p = parents[i]
        if p in childs:
            childs[p].append(i)
        else:
            childs[p] = [i]
        if p in des:
            des[p].append(i)
        else:
            des[p] = [i]
    for i in refs:
        for p in refs[i]:
            if p in des:
                des[p].append(i)
            else:
                des[p] = [i]
    print("Finish construct childs", len(childs), len(des))
    subtree = {}
    subtree_weight = {}
    for start in g_time:
        weight = 0
        queue = Queue()
        queue.put(start)
        accessed = {}
        subtree[start] = []
        while not queue.empty():
            b = queue.get()
            if b in accessed:
                continue
            if b in childs:
                for c in childs[b]:
                    queue.put(c)
            accessed[b] = True
            weight += difficulties[b]
            if b != start:
                subtree[start].append(b)
        subtree_weight[start] = weight
    print("Finish construct subtree", len(subtree))
    future = {}
    # for start in g_time:
    #     queue = Queue()
    #     queue.put(start)
    #     accessed = {}
    #     future[start] = []
    #     while not queue.empty():
    #         b = queue.get()
    #         if b in accessed:
    #             continue
    #         if b in des:
    #             for c in des[b]:
    #                 queue.put(c)
    #         accessed[b] = True
    #         if b != start:
    #             future[start].append(b)
    # print("Finish construct future", len(future))
    c_time = {}
    for b in aim_blocks:
        # print("generate_time", b, g_time[b])
        if b not in parents:
            print("skip not in parents", b)
            continue
        if parents[b] not in r_time:
            print("skip not in r_time", parents[b])
            continue
        # print("subtree", len(subtree[b]))
        subtree_sorted = sorted(
            [_ for _ in subtree[b] if _ in r_time], key=lambda x: r_time[x])
        siblings = [_ for _ in childs[parents[b]] if _ != b]
        weight = 0
        # print("sorted", len(subtree_sorted))
        sib_tree_size = 0
        if len(siblings) != 0:
            sib_tree_size = max([subtree_weight[sib]
                                 for sib in siblings])
        # print("sib", sib_tree_size)
        for j in range(len(subtree_sorted)):
            r_t = r_time[subtree_sorted[j]]
            weight += difficulties[subtree_sorted[j]]
            # print(weight, sib_tree_size)
            if accept(
                    (r_t - r_time[parents[b]]), lambda_n, sib_tree_size, weight, r):
                c_time[b] = r_t
                # print("confirm_time", r_t)
                break
        # print("weight", weight)
    final_c_time = {}
    for b in aim_blocks:
        if b in c_time:
            final_c_time[b] = c_time[b]
        if b in future:
            f_commit = [c_time[_] for _ in future[b] if _ in c_time]
            if len(f_commit) == 0:
                continue
            else:
                final_c_time[b] = min(f_commit)
    for b in final_c_time:
        if b in future:
            for f in future[b]:
                if f in final_c_time and final_c_time[f] < final_c_time[b]:
                    final_c_time[f] = final_c_time[b]
    lat = []
    # for b in c_time:
    #     lat.append((c_time[b] - g_time[b]))
    for b in final_c_time:
        lat.append((final_c_time[b] - g_time[b]))
    lat_s = sorted(lat)
    print("Latency from block number: ", len(lat_s))
    # print("Latency is %.2f" % lat_s[0])
    print("Latency is %.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (lat_s[0], lat_s[int(len(
        lat_s) * 0.25)], sum(lat_s) / len(lat_s), lat_s[int(len(lat_s) * 0.75)],
        lat_s[-1]))
    return lat_s
