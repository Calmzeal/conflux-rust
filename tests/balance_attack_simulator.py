#!/usr/bin/env python3
import queue
import random
import multiprocessing
from statistics import mean
import time

class Environment:

    def __init__(self):
        return


class AbstractNode:

    def __init__(self,num_nodes):
        self.ltree = 0
        self.rtree = 0
        self.received = {}


class Simulator:

    def __init__(self, env, threshold):
        self.e = env
        # environment checker
        for attr in ["num_nodes","total_period","evil_rate","latency","topo","termination_time"]:
            if not hasattr(self.e,  attr):
                print("{} unsetted".format(attr))
                exit()

        self.threshold = threshold
        self.history = queue.PriorityQueue()

    def setup_chain(self):
        self.nodes = []
        for i in range(self.e.num_nodes):
            self.nodes.append(AbstractNode(self.e.num_nodes))
        self.ltree = 0
        self.rtree = 0
        self.lq = queue.Queue()
        self.rq = queue.Queue()
        return

    def setup_network(self):
        self.g = []
        for i in range(self.e.num_nodes):
            self.g.append([])
            for j in range(self.e.topo):
                seed = random.randint(0, self.e.num_nodes-1)
                while seed in self.g[i] or seed == i:
                    seed = random.randint(0, self.e.num_nodes-1)
                self.g[i].append(seed)
        return

    def run_test(self):

        # Initialize the target's tree
        ltarget = list(range(0, self.e.num_nodes, 2))
        rtarget = list(range(1, self.e.num_nodes, 2))
        for i in ltarget:
            self.nodes[i].ltree = 0
        for i in rtarget:
            self.nodes[i].rtree = 1

        # Executed the simulation
        miner = 0
        timestamp = 0
        while timestamp < self.e.termination_time:
            timestamp += random.expovariate(1 / self.e.total_period)
            #miner = random.randint(0, self.e.num_nodes-1)
            self.parse_mining_history(timestamp)

            if random.random() < self.e.evil_rate:
                # Check termination
                merged_to_left = True
                merged_to_right = True
                for i in range(self.e.num_nodes):
                    merged_to_left &= self.nodes[i].ltree >= self.nodes[i].rtree
                    merged_to_right &= self.nodes[i].rtree > self.nodes[i].ltree
                if merged_to_left or merged_to_right:
                    print("Chain merged after {} seconds".format(timestamp))
                    return timestamp
                # # Decide attack target
                q,chirality,target = (self.lq, "L",ltarget) if self.lq.qsize() + self.ltree < self.rq.qsize() + self.rtree else (self.rq, "R", rtarget)
                q.put(timestamp)
                send_count = abs(self.ltree-self.rtree)
                if send_count >= self.threshold:
                    for i in range(send_count):
                        if q.empty():
                            break
                        blk = q.get()
                        if chirality == "L":
                            self.ltree += 1
                        else:
                            self.rtree += 1
                        for j in target:
                            self.history.put((timestamp+random.randrange(0,1), j, chirality, blk))
            else:
                miner = (miner + 1) % (self.e.num_nodes - 1)
                #miner = random.randint(0, self.e.num_nodes-1)
                chirality = "L" if self.nodes[miner].ltree >= self.nodes[miner].rtree else "R"
                # Update attacker and miner's views
                self.nodes[miner].received.update({timestamp:True})
                if chirality == "L":
                    self.nodes[miner].ltree += 1
                    self.ltree += 1
                else:
                    self.nodes[miner].rtree += 1
                    self.rtree += 1
                # Broadcast new blocks to neighbours
                self.broadcast(timestamp, miner, chirality, timestamp)

        print("Chain unmerged after {} seconds... ".format(self.e.termination_time))
        return self.e.termination_time


    def broadcast(self, origin_time, index, chirality, blk):
        for i in range(self.e.topo):
            new_stamp = origin_time + self.e.latency # + random.uniform(0, 1)
            self.history.put((new_stamp, self.g[index][i], chirality, blk))


    def parse_mining_history(self, current_stamp):
        # Parse events and generate new ones in a BFS way
        while True:
            # Safely get valid event from history
            if self.history.empty():
                return
            stamp, index, chirality, blk = self.history.get()
            if stamp > current_stamp:
                self.history.put((stamp, index, chirality, blk))
                return

            # Only new blocks will modify the memory
            if self.nodes[index].received.get(blk) is None:
                self.nodes[index].received.update({blk:True})
                # Update the subtree weight
                if chirality == "L":
                    self.nodes[index].ltree += 1
                else:
                    self.nodes[index].rtree += 1
                self.broadcast(stamp, index, chirality, blk)

    def main(self):
        self.setup_chain()
        self.setup_network()
        return self.run_test()



def slave_simulator(auxiliary):
    env = Environment()
    env.num_nodes = 100
    env.total_period = 0.25
    env.evil_rate = 0.2
    env.latency = 10
    env.topo = 5
    env.termination_time = 6000

    return Simulator(env,3).main()

if __name__=="__main__":


    cpu_num = multiprocessing.cpu_count()-15
    p = multiprocessing.Pool(cpu_num)
    begin = time.time()
    repeats = 100
    attack_last_time = sorted(p.map(slave_simulator, [0]*(repeats+1)))
    samples = 10
    print("len: %s" % len(attack_last_time))
    print(list(map(lambda percentile: attack_last_time[int((repeats - 1) * percentile / samples)], range(samples + 1))))
    end = time.time()
    print("Executed in {} seconds".format(end-begin))
