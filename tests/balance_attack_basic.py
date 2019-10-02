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
            "test_mining_sleep_us": "10000",
            "mining_author": '"' + "0" * 40 + '"',
            "log_level": "\"debug\"",
            "headers_request_timeout_ms": "30000",  # need to be larger than network latency [What is it?]
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
        start_p2p_connection(self.nodes)
        generation_period = 1 / (1 / self.total_period * self.evil_rate)
        self.log.info("Adversary mining average period=%f", generation_period)

        # Some blocks may have been mined before we setup the latency, [which code corresponds to the decalaration?]
        # so wait for the latency and find the fork point, [which code corresponds to the decalaration?]
        ''' Phase 1: Scan the chain every 0.1s until find a fork at a valid height
            between nodes[0] and nodes[branch_leader]
        '''
        finished = False
        branch_leader = int(self.num_nodes / 2)
        while not finished:
            chain0 = self._process_chain(self.nodes[0].getPivotChainAndWeight())
            chain1 = self._process_chain(self.nodes[branch_leader].getPivotChainAndWeight())
            fork_height = 0
            while True:
                if chain0[fork_height][0] != chain1[fork_height][0]:
                    fork0 = chain0[fork_height]
                    fork1 = chain1[fork_height]
                    self.log.info("Forked at height %d %s %s", fork_height, fork0, fork1)
                    finished = True
                    break
                fork_height += 1
                if fork_height >= min(len(chain0), len(chain1)):
                    self.log.info("No fork to start attack, retry")
                    time.sleep(0.1)
                    break
        if fork_height >= 5:
            # Our generated block has height fork_height+1, so its deferred block is fork_height-4
            receipts_root = decode_hex(self.nodes[0].getExecutedInfo(chain0[fork_height - 4][0])[0])
        else:
            receipts_root = default_config["GENESIS_RECEIPTS_ROOT"]

        ''' Phase 2: At each given adversary block generation moment (decided by adversary's mining power),
            send blocks to each fork until they merge and are stabilized. 
        '''
        count = 0
        after_count = 0
        merged = False
        while True:
            sample0 = randint(0, branch_leader - 1)
            sample1 = randint(branch_leader, self.num_nodes - 1)
            # This roughly simulates adversary's mining power
            time.sleep(random.expovariate(1 / generation_period))
            chain0 = self._process_chain(self.nodes[sample0].getPivotChainAndWeight())
            self._check_chain_heavy(chain0, 0, fork_height)
            chain1 = self._process_chain(self.nodes[sample1].getPivotChainAndWeight())
            self._check_chain_heavy(chain1, 1, fork_height)
            assert_equal(chain0[0][0], chain1[0][0])
            fork0 = chain0[fork_height]
            fork1 = chain1[fork_height]
            self.log.debug("Fork root %s %s", chain0[fork_height], chain1[fork_height])
            if fork0[0] == fork1[0]:
                merged = True
                self.log.info("Pivot chain merged")
                # self.log.info("chain0 %s", chain0)
                # self.log.info("chain1 %s", chain1)
                after_count += 1
                if after_count >= 120 / generation_period:
                    self.log.info("Merged. Winner: %s Chain end with %s", fork0[0],
                                  chain0[min(len(chain0), len(chain1)) - 2][0])
                    break
                continue

            count += 1
            if count >= 2400 / generation_period:
                self.log.info("Not merged after 40 min")
                break
            ''' Send blocks to keep balance.
                The adversary's mining power and strategy is not strictly designed in the naive version.
                If two forks are already balanced, we need to send blocks to both sides in case no blocks are mined.
            '''
            if self.start_attack:
                if not merged:
                    if fork0[1] < fork1[1] or (fork0[1] == fork1[1] and fork0[0] < fork1[0]):
                        send1 = True
                    else:
                        send1 = False
                if send1:

                    parent = fork0[0]
                    block = NewBlock(
                        create_block(decode_hex(parent), height=fork_height + 1, deferred_receipts_root=receipts_root,
                                     difficulty=self.difficulty, timestamp=int(time.time()),
                                     author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
                    for i in range(branch_leader):
                        self.nodes[i].p2p.send_protocol_msg(block)
                    self.log.info("send to 0 group block %s, weight %d %d", block.block.hash_hex(), fork0[1], fork1[1])
                else:
                    parent = fork1[0]
                    block = NewBlock(
                        create_block(decode_hex(parent), height=fork_height + 1, deferred_receipts_root=receipts_root,
                                     difficulty=self.difficulty, timestamp=int(time.time()),
                                     author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
                    for i in range(branch_leader, self.num_nodes):
                        self.nodes[i].p2p.send_protocol_msg(block)
                    self.log.info("send to 1 group block %s, weight %d %d", block.block.hash_hex(), fork0[1], fork1[1])
        exit()

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


if __name__ == "__main__":
    P2PTest().main()
