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
        self.num_nodes = 4
        self.start_attack = True
        self.branch_leader = int(self.num_nodes / 2)
        self.fork_hash_list = []

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
        ''' Phase 1: Scan the chain every 0.1s until find a fork at a valid height
            between nodes[0] and nodes[branch_leader]
        '''
        fork_height,receipts_root = self._getfork()
        print(self.fork_hash_list)
        ''' Phase 2: At each given adversary block generation moment (decided by adversary's mining power),
            send blocks to each fork until they merge and are stabilized. 
        '''
        branch_leader = self.branch_leader
        count = 0
        after_count = 0
        merged = False
        weights_old = [0]*self.num_nodes
        while True:
            # This roughly simulates adversary's mining power
            time.sleep(random.expovariate(1 / generation_period))
            # chain0 = self._process_chain(self.nodes[0].getPivotChainAndWeight())
            # self._check_chain_heavy(chain0, 0, fork_height)
            # chain1 = self._process_chain(self.nodes[1].getPivotChainAndWeight())
            # self._check_chain_heavy(chain1, 1, fork_height)
            # assert_equal(chain0[0][0], chain1[0][0])
            # fork0 = chain0[fork_height]
            # fork1 = chain1[fork_height]
            comparable = True
            stable = True
            weights = []

            for i in range(self.num_nodes):
                tmp = self.nodes[i].weight.get(self.fork_hash_list[i])
                if tmp is None:
                    comparable = False
                    break
                weights.append(tmp)
                if weights[i] !=  weights_old[i]:
                    stable = False
            if comparable:
                print(weights)
            else:
                continue

            weights_old = weights

            if stable:
               after_count = after_count + 1
            else:
                if after_count > 600:
                    merged = True
                    self.log.info("Pivot chain merged")
                    break
                else:
                    after_count = 0
            count = count + 1
            if count > 2400:
                self.log.info("Not merged after 40 min")
                break

            # print(weights)
            # print(fork0,fork1)
            # self.log.debug("Fork root %s %s", chain0[fork_height], chain1[fork_height])
            # if fork0[0] == fork1[0]:
            #     merged = True
            #     self.log.info("Pivot chain merged")
            #     self.log.info("chain0 %s", chain0)
            #     self.log.info("chain1 %s", chain1)
            #     after_count += 1
            #     if after_count >= 120 / generation_period:
            #         self.log.info("Merged. Winner: %s Chain end with %s", fork0[0],
            #                       chain0[min(len(chain0), len(chain1)) - 2][0])
            #         break
            #     continue
            # count += 1
            # if count >= 2400 / generation_period:
            #     self.log.info("Not merged after 40 min")
            #     break
            ''' Send blocks to keep balance.
                The adversary's mining power and strategy is not strictly designed in the naive version.
                If two forks are already balanced, we need to send blocks to both sides in case no blocks are mined.
            '''

            if self.start_attack:
                target = min(weights)
                for i in range(self.num_nodes):
                    if weights[i]==target:
                        parent = self.fork_hash_list[i]
                        block = NewBlock(
                            create_block(decode_hex(parent), height=fork_height + 1, deferred_receipts_root=receipts_root,
                                     difficulty=self.difficulty, timestamp=int(time.time()),
                                     author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
                        self.log.info("send to %d group block %s", i, block.block.hash_hex())
                        break


            # if self.start_attack:
            #     if not merged:
            #         if fork0[1] < fork1[1] or (fork0[1] == fork1[1] and fork0[0] < fork1[0]):
            #             send1 = True
            #         else:
            #             send1 = False
            #     if send1:
            #         parent = fork0[0]
            #         block = NewBlock(
            #             create_block(decode_hex(parent), height=fork_height + 1, deferred_receipts_root=receipts_root,
            #                          difficulty=self.difficulty, timestamp=int(time.time()),
            #                          author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
            #         for i in range(branch_leader):
            #             self.nodes[i].p2p.send_protocol_msg(block)
            #         # self.log.info("send to 0 group block %s, weight %d %d", block.block.hash_hex(), fork0[1], fork1[1])
            #     else:
            #         parent = fork1[0]
            #         block = NewBlock(
            #             create_block(decode_hex(parent), height=fork_height + 1, deferred_receipts_root=receipts_root,
            #                          difficulty=self.difficulty, timestamp=int(time.time()),
            #                          author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
            #         for i in range(branch_leader, self.num_nodes):
            #             self.nodes[i].p2p.send_protocol_msg(block)
                    # self.log.info("send to 1 group block %s, weight %d %d", block.block.hash_hex(), fork0[1], fork1[1])
        exit()

    def _getfork(self):
        ''' snapshot(non-atomic) the pivot chain of each cfx node,
            scan the snapshot to find the fork height
        :return: fork_height, receipt at the fork point.
        '''
        fork_found = False
        branch_leader = self.branch_leader
        self.fork_hash_list = []
        chain = []
        while not fork_found:
            chain = []
            for i in range(self.num_nodes):
                chain.append(self._process_chain(self.nodes[i].getPivotChainAndWeight()))
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
                        self.log.info("Forked at height %d %s %s", height, chain[i][height],chain[i+1][height])
                        for j in range(self.num_nodes):
                            self.fork_hash_list.append(chain[j][height][0])
                        if height >= 5:
                            # Our generated block has height fork_height+1, so its deferred block is fork_height-4
                            return height, decode_hex(self.nodes[0].getExecutedInfo(chain[0][height - 4][0])[0])
                        else:
                            return height, default_config["GENESIS_RECEIPTS_ROOT"]
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


if __name__ == "__main__":
    P2PTest().main()
