#!/usr/bin/env python3

""" 2019-10-01
    Log :
    @code.last_version

    Libraries introduction:

    "argparse" - help rpc parse
    http - connection
    eth, conflux - block info storage <> comm message typing changer
    test_framework - including tools triggering each phase in blockchain network establishment
"""
import os
import queue
from argparse import ArgumentParser
from threading import Thread

from conflux.utils import parse_as_int
from test_framework.blocktools import create_block
from test_framework.test_framework import ConfluxTestFramework
from test_framework.mininode import *
from test_framework.util import *

from tests.conflux.rpc import RpcClient
from eth_utils import encode_hex, decode_hex


class P2PTest(ConfluxTestFramework):
    """ Def P2PTest [X : <BlockchainType>TestFramework -> Y: Class(_Auxiliary, f1, f2, f3)]
        A 3 phase attacking test over a given blockchain test framework,
        including setup_chain, setup_network, and run_test.
    """
    def set_test_params(self):
        self.setup_clean_chain = True
        self.num_nodes = 6
        self.start_attack = True
        self.target = AttackTarget(self.num_nodes)
        self.task_queue = TaskQueue()

    def add_options(self, parser: ArgumentParser):
        parser.add_argument(
            "--evil",
            dest="evil_rate",
            # Modifiable: the mining power of the adversary part
            default=0.2,
            type=float,
        )

    def setup_chain(self):
        self.log.info("Initializing test directory " + self.options.tmpdir)

        self.total_period = 0.25
        self.evil_rate = self.options.evil_rate
        self.difficulty = int(self.num_nodes / (1 / self.total_period) / (1 - self.evil_rate) * 100)
        self.target.difficulty = self.difficulty

        print(self.difficulty)
        self.conf_parameters = {
            "start_mining": "true",
            "initial_difficulty": str(self.difficulty),
            "test_mining_sleep_us": "10000",  # mining might have happened before the network is set up.
            "mining_author": '"' + "0" * 40 + '"',
            "log_level": "\"debug\"",
            "headers_request_timeout_ms": "30000",  # need to be larger than network latency
            "heavy_block_difficulty_ratio": "1000",  # parameter used in the original experiments
            "adaptive_weight_beta": "3000",  # parameter used in the original experiments
        }
        self._initialize_chain_clean()

    def setup_network(self):
        self.setup_nodes()
        ''' Set the topology of the network: complete graph;
            Then set the latency over the topology
        '''
        self.log.info("Connection Started")
        for i in range(self.num_nodes - 1):
            for j in range(i + 1, self.num_nodes):
                connect_nodes(self.nodes, i, j)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    self.nodes[i].addlatency(self.nodes[j].key, 10000)
        self.log.info("Connection Finished")

    def run_test(self):
        # Connect mininodes to cfx nodes
        start_p2p_connection(self.nodes,False,self.task_queue,self.target) # remote = False

        generation_period = 1 / (1 / self.total_period * self.evil_rate)
        self.log.info("Adversary mining average period=%f", generation_period)

        self.set_target()

        # Executed the attacks
        count = 0
        after_count = 0
        scan_freq = 20
        threshhold = 0
        merged = False
        t = self.target
        lblock_queue = queue.Queue()
        rblock_queue = queue.Queue()
        chain = []

        while self.start_attack:
            # This roughly simulates adversary's mining power
            time.sleep(random.expovariate(1 / generation_period))

            # Get the subtree weight of the two branches
            _,lweight,rweight = t.get_weights(t.lhash,t.rhash)
            self.log.info("Left:{}, Right:{}, Delta:{} #Lqueue：{}, Rqueue:{} #Downloading blocks:{}" \
                          .format(lweight, rweight, abs(lweight - rweight),lblock_queue.qsize(),rblock_queue.qsize(), len(t.weight)))

            # Check test's termination
            count += 1
            if count > 2400 / generation_period:
                self.log.info("Not merged after 40 minutes")
                break

            # Check merge every 100 period of attack
            if count%scan_freq == 0:
                chain = self.get_chains()

                weights = list(map(lambda x:t.weight.get(chain[x][t.fork_height][0]),range(self.num_nodes)))
                values = set(weights)
                print(values)
                if len(values) <= 2:
                    threshhold = 4
                merged = True
                for i in range(self.num_nodes-1):
                    merged &= chain[i][t.fork_height][0] == chain[i+1][t.fork_height][0]

                #self.check_chain_heavy(chain[0], 0, t.fork_height)
                if merged:
                    self.log.info("Pivot chain merged")
                    scan_freq = 1
                    after_count += 1
                    if after_count >= 20 / generation_period:
                        self.log.info("Merged. Winner: %s among [%s,%s]", chain[0][t.fork_height][0],t.lhash,t.rhash)
                        break
                elif after_count > 0:
                    after_count = 0
                    scan_freq = 5

            # Attack even pivot chain are merged
            to_left = lweight + lblock_queue.qsize() <= rweight + rblock_queue.qsize()
            parent, q = (t.lhash, lblock_queue) if to_left else (t.rhash, rblock_queue)
            q.put(NewBlock(
                create_block(decode_hex(parent), height=t.fork_height + 1, deferred_receipts_root=t.receipts_root,
                             difficulty=self.difficulty, timestamp=int(time.time()),
                             author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1)))))

            send_count = (abs(lweight - rweight) + int((1 - self.evil_rate) / self.evil_rate)) if merged else abs(lweight - rweight)
            if send_count>=threshhold:
                group, targetlist = ("right", t.rtarget) if (lweight-rweight>0) else ("left", t.ltarget)
                if merged:
                    group, targetlist = ("right", t.rtarget) if (t.rhash != chain[0][t.fork_height][0]) else ("left", t.ltarget)
                for k in range(send_count):
                    if q.empty():
                        break
                    blk = q.get()
                    for i in targetlist:
                        self.nodes[i].p2p.send_protocol_msg(blk)
                    self.log.info("send to %s group block %s", group, blk.block.hash_hex())

        os.system("killall conflux")
        exit()




    def get_chains(self):
        '''
        :return: list of pivot chain from different cfx nodes
        '''
        chain = []
        for i in range(self.num_nodes):
            chain.append(self._process_chain(self.nodes[i].getPivotChainAndWeight()))
        return chain

    def set_target(self):
        ''' snapshot(non-atomic) the pivot chain of each cfx node,
            scan the snapshot to find the fork height,
            then set self.target
        '''
        while True:
            chain = self.get_chains()
            height = 0
            finished = False
            while not finished:
                finished = False
                for i in range(self.num_nodes):
                    if height >= len(chain[i]):
                        self.log.info("No fork to start attack, retry")
                        time.sleep(0.1)
                        finished = True
                        break
                if finished:
                    break
                for i in range(self.num_nodes-1):
                    if chain[i][height][0] != chain[i+1][height][0]:
                        lhash = chain[i][height][0]
                        rhash = chain[i+1][height][0]
                        time.sleep(25)
                        received,l,r = self.target.get_weights(lhash,rhash)
                        if received:
                            self.log.info("Forked at height %d %s %s", height, chain[i][height], chain[i + 1][height])
                            # Our generated block has height fork_height+1, so its deferred block is fork_height-4
                            receipts_root = decode_hex(self.nodes[i].getExecutedInfo(chain[i][height - 4][0])[0])\
                                if height >= 5 else default_config["GENESIS_RECEIPTS_ROOT"]
                            self.target.set_params(i,lhash,rhash,height,receipts_root)
                            return None
                        else:
                            print(l,r)
                            self.log.info("Fork block unreceived... retry")
                            finished = True
                            break
                height += 1


    # Convert weight to integer
    def _process_chain(self, chain):
        for i in range(len(chain)):
            chain[i][1] = parse_as_int(chain[i][1])
        return chain

    def check_chain_heavy(self, chain, chain_id, fork_height):
        for i in range(fork_height + 1, len(chain) - 1):
            if chain[i][1] - chain[i + 1][1] >= self.difficulty * 240:
                self.log.info("chain %d is heavy at height %d %d %d", chain_id, i, chain[i][1], chain[i + 1][1])
                return
        if chain[-1][1] >= self.difficulty * 240:
            self.log.info("chain %d is heavy at height %d %d %d", chain_id, i, chain[i][1], chain[i + 1][1])


class AttackTarget:
    """ Define the scale of the network, the attack strategy and parameters
    """
    def __init__(self,num_nodes):
        self.ltarget = list(range(0, num_nodes, 2))
        self.rtarget = list(range(1, num_nodes, 2))
        # The following dictionaries store the received new blocks and construct the global tree graph
        def f(t, node, new_block_hashes):
            for h in new_block_hashes:
                x = RpcClient(node).block_by_hash(encode_hex(h))
                difficulty = int(x['difficulty'],0)
                if difficulty > t.difficulty * 240:
                    print("Heavy block comes from node {} with difficulty {}".format(node.index, difficulty))
                relative_weight = int(difficulty/t.difficulty)
                if t.parent_map.get(x["hash"]) is None:
                    original_weight = t.weight.get(x["hash"])
                    if original_weight is None:
                        t.weight.update({x["hash"]: relative_weight})
                        original_weight = relative_weight
                    t.parent_map.update({x["hash"]: x["parentHash"]})
                    child = x["hash"]
                    parent = t.parent_map.get(child)
                    parent_weight = t.weight.get(parent)
                    if parent_weight is None:
                        t.weight.update({parent: original_weight})
                    else:
                        while not (parent is None):
                            parent_weight += original_weight
                            t.weight.update({parent: parent_weight})
                            child = parent
                            parent = t.parent_map.get(child)
                            parent_weight = t.weight.get(parent)

        self.parse_block_hashes = f
        self.parent_map = {}
        self.weight = {}

    def set_params(self,branch_leader,lhash,rhash,fork_height,receipts_root):
        self.branch_leader = branch_leader
        self.lhash = lhash
        self.rhash = rhash
        self.fork_height = fork_height
        self.receipts_root = receipts_root

        if branch_leader in self.rtarget:
            self.rtarget.remove(branch_leader)
            self.ltarget.append(branch_leader)
        if (branch_leader + 1) in self.ltarget:
            self.ltarget.remove(branch_leader + 1)
            self.rtarget.append(branch_leader + 1)

    def get_weights(self,lhash,rhash):
        '''
        :return: received:bool, lweight, rweight
        '''
        lweight = self.weight.get(lhash)
        rweight = self.weight.get(rhash)
        return not ((lweight is None) | (rweight is None)),lweight,rweight

class TaskQueue(queue.Queue):
    """ Single Consumer Task Queue Class Dealing with Nodes
    """
    def __init__(self):
        super().__init__()
        self.start()

    def add_task(self,task, *args, **kwargs):
        self.put((task, args, kwargs))

    def start(self):
        Thread(target=self.consumer, daemon=True).start()

    def consumer(self):
        while True:
            item, args, kwargs = self.get()
            item(*args, **kwargs)
            self.task_done()

if __name__ == "__main__":
    P2PTest().main()
