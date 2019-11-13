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

from random import randint
from argparse import ArgumentParser
from eth_utils import decode_hex
from conflux.utils import parse_as_int
from test_framework.blocktools import create_block
from test_framework.test_framework import ConfluxTestFramework
from test_framework.mininode import *
from test_framework.util import *


class P2PTest(ConfluxTestFramework):
    """ Def P2PTest [X : <BlockchainType>TestFramework -> Y: Class(_Auxiliary, f1, f2, f3)]
        A 3 phase attacking test over a given blockchain test framework,
        including setup_chain, setup_network, and run_test.
    """
    def set_test_params(self):
        self.setup_clean_chain = True
        self.num_nodes = 2
        self.start_attack = True
        # branch_leader is calculated in _getfork_and_sethash() according to definition:
        # 1. node[0..branch_leader][fork_height] are equal
        # 2. node[branch_leader][fork_height] != node[branch_leader+1]
        # We group the nodes in [0..branch_leader]+several nodes as group 1 and others as group 2
        self.branch_leader = None
        self.lhash = None
        self.rhash = None
        # task_queue store the received new blocks to construct the global tree graph
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
        # Modifiable: the expected block generation rate for the whole chain
        self.total_period = 0.25
        self.evil_rate = self.options.evil_rate
        # Modifiable: v.Peilun 2 -> v.Enlin num_nodes
        self.difficulty = int(self.num_nodes / (1 / self.total_period) / (1 - self.evil_rate) * 100)
        ''' Exp Proposition : The higher the difficulty is, the less probable to fork
        '''
        # self.difficulty = 75
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
        #Connect mininodes to cfx nodes
        start_p2p_connection(self.nodes,False,self.task_queue)

        generation_period = 1 / (1 / self.total_period * self.evil_rate)
        self.log.info("Adversary mining average period=%f", generation_period)

        # Scan the pivot chains to get the fork height and store the two branches' hashes
        fork_height, receipts_root = self._getfork_and_sethash()

        # Set the attacking groups
        cnt = 0
        ltarget = [self.branch_leader]
        rtarget = [self.branch_leader+1]
        for i in range(self.num_nodes):
            if i == self.branch_leader or i == self.branch_leader+1:
                continue
            cnt += 1
            if cnt < (self.num_nodes / 2):
                ltarget.append(i)
            else:
                rtarget.append(i)
        print(ltarget, rtarget)

        # Executed the attacks
        count = 0
        after_count = 0
        merged = False
        while self.start_attack:
            # This roughly simulates adversary's mining power
            time.sleep(random.expovariate(1 / generation_period))
            # Get the subtree weight of the two branches
            comparable, lweight, rweight = self._getweights()
            if comparable:
                print(lweight,rweight)
                print(len(self.task_queue.parent_map))
            else:
                print(len(self.task_queue.parent_map))
                continue

            # Check test's termination
            count = count + 1
            if count > 2400:
                self.log.info("Not merged after 40 min")
                break

            # Check merge every 100 period of attack
            if count%100 == 0:
                chain = self._snapshot_chain()
                merged = True
                for i in range(self.num_nodes-1):
                    merged &= chain[i][fork_height][0] == chain[i+1][fork_height][0]
                self._check_chain_heavy(chain[0], 0, fork_height)
                if merged:
                    self.log.info("Pivot chain merged")
                    after_count += 1
                    if after_count >= 3 / generation_period:
                        self.log.info("Merged. Winner: %s among [%s,%s]", chain[0][fork_height][0],self.lhash,self.rhash)
                        break

            # Attack
            if not merged:
                to_left = lweight <= rweight
                parent,target = (self.lhash, ltarget) if to_left else (self.rhash, rtarget)
                block = NewBlock(
                    create_block(decode_hex(parent), height=fork_height + 1, deferred_receipts_root=receipts_root,
                                 difficulty=self.difficulty, timestamp=int(time.time()),
                                 author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
                for i in target:
                    self.nodes[i].p2p.send_protocol_msg(block)
                self.log.info("send to %s group block %s", parent, block.block.hash_hex())
        exit()

    def _getweights(self):
        '''
        :return: comparable:bool, lweight, rweight
        '''
        q = self.task_queue
        lweight = q.weight.get(self.lhash)
        rweight = q.weight.get(self.rhash)
        return not ((lweight is None) | (rweight is None)),lweight,rweight

    def _snapshot_chain(self):
        chain = []
        for i in range(self.num_nodes):
            chain.append(self._process_chain(self.nodes[i].getPivotChainAndWeight()))
        return chain

    def _getfork_and_sethash(self):
        ''' snapshot(non-atomic) the pivot chain of each cfx node,
            scan the snapshot to find the fork height
        :return: fork_block_hashes, fork_height, receipt at the fork point.
        '''
        while True:
            chain = self._snapshot_chain()
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
                        self.branch_leader = i
                        self.lhash = chain[i][height][0]
                        self.rhash = chain[i+1][height][0]
                        time.sleep(20)
                        comparable, lweight, rweight = self._getweights()
                        if comparable:
                            self.log.info("Forked at height %d %s %s", height, chain[i][height], chain[i + 1][height])
                            # Our generated block has height fork_height+1, so its deferred block is fork_height-4
                            receipts_root = decode_hex(self.nodes[i].getExecutedInfo(chain[i][height - 4][0])[0])\
                                if height >= 5 else default_config["GENESIS_RECEIPTS_ROOT"]
                            return height, receipts_root
                        else:
                            print(lweight,rweight)
                            self.log.info("Fork block unreceived... retry")
                            finished = True
                            break
                height += 1


    # Convert weight to integer
    def _process_chain(self, chain):
        for i in range(len(chain)):
            chain[i][1] = parse_as_int(chain[i][1])
        return chain

    def _check_chain_heavy(self, chain, chain_id, fork_height):
        for i in range(fork_height + 1, len(chain) - 1):
            if chain[i][1] - chain[i + 1][1] >= self.difficulty * 240:
                self.log.info("chain %d is heavy at height %d %d %d", chain_id, i, chain[i][1], chain[i + 1][1])
                return
        if chain[-1][1] >= self.difficulty * 240:
            self.log.info("chain %d is heavy at height %d %d %d", chain_id, i, chain[i][1], chain[i + 1][1])

class TaskQueue(queue.Queue):
    """ Single Consumer Task Queue Class Dealing with Nodes
    """
    def __init__(self):
        super().__init__()
        self.parent_map = {}
        self.weight = {}
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
