#!/usr/bin/env python3
from http.client import CannotSendRequest

from eth_utils import decode_hex

from conflux.utils import encode_hex, privtoaddr, parse_as_int
from test_framework.blocktools import create_transaction, create_block
from test_framework.test_framework import ConfluxTestFramework
from test_framework.mininode import *
from test_framework.util import *
from queue import Queue

class P2PTest(ConfluxTestFramework):
    def set_test_params(self):
        self.setup_clean_chain = True
        self.num_nodes = 2
        self.start_attack = True
        self.difficulty = 1000
        self.conf_parameters = {
            "start_mining": "true",
            "initial_difficulty": str(self.difficulty),
            "test_mining_sleep_us": "100",
            "mining_author": '"' + "0"*40 + '"',
        }

    def setup_network(self):
        self.setup_nodes()
        connect_nodes(self.nodes, 0, 1)

        # Set latency between two groups
        self.nodes[0].addlatency(self.nodes[1].key, 1000)
        self.nodes[1].addlatency(self.nodes[0].key, 1000)

    def run_test(self):
        start_p2p_connection(self.nodes)

        # Some blocks may have been mined before we setup the latency,
        # so wait for the latency and find the fork point
        finished = False
        while not finished:
            chain0 = self.process_chain(self.nodes[0].getPivotChainAndWeight())
            chain1 = self.process_chain(self.nodes[1].getPivotChainAndWeight())
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
                    break
        if fork_height >= 5:
            # Our generated block has height fork_height+1, so its deferred block is fork_height-4
            receipts_root = decode_hex(self.nodes[0].getExecutedInfo(chain0[fork_height - 4][0])[0])
        else:
            receipts_root = default_config["GENESIS_RECEIPTS_ROOT"]

        block_set0 = Queue()
        block_set1 = Queue()
        generation_period = 0.1
        mining_thread0 = AttackMiningThread(fork0[0], fork_height, block_set0, receipts_root, self.difficulty, generation_period * 2)
        mining_thread1 = AttackMiningThread(fork1[0], fork_height, block_set1, receipts_root, self.difficulty, generation_period * 2)
        mining_thread0.start()
        mining_thread1.start()
        while True:
            # This roughly determines adversary's mining power

            chain0 = self.process_chain(self.nodes[0].getPivotChainAndWeight())
            self.check_chain_heavy(chain0, 0, fork_height)
            chain1 = self.process_chain(self.nodes[1].getPivotChainAndWeight())
            self.check_chain_heavy(chain1, 1, fork_height)
            assert_equal(chain0[0][0], chain1[0][0])
            fork0 = chain0[fork_height]
            fork1 = chain1[fork_height]
            self.log.debug("Fork weight %d %d", fork0[1], fork1[1])
            if fork0[0] == fork1[0]:
                self.log.info("Pivot chain merged")
                self.log.info("chain0 %s", chain0)
                self.log.info("chain1 %s", chain1)
                break

            ''' Send blocks to keep balance.
                The adversary's mining power and strategy is not strictly designed in the naive version.
                If two forks are already balanced, we need to send blocks to both sides in case no blocks are mined.
            '''
            if self.start_attack:
                if fork0[1] <= fork1[1]:
                    try:
                        block = block_set0.get_nowait()
                        self.nodes[0].p2p.send_protocol_msg(block)
                        self.log.info("send to 0 block %s, weight %d %d", block.block.hash_hex(), fork0[1], fork1[1])
                    except Exception:
                        pass
                if fork0[1] >= fork1[1]:
                    try:
                        block = block_set1.get_nowait()
                        self.nodes[1].p2p.send_protocol_msg(block)
                        self.log.info("send to 1 block %s, weight %d %d", block.block.hash_hex(), fork0[1], fork1[1])
                    except Exception:
                        pass
        exit()
    # Convert weight to integer
    def process_chain(self, chain):
        for i in range(len(chain)):
            chain[i][1] = parse_as_int(chain[i][1])
        return chain

    def check_chain_heavy(self, chain, chain_id, fork_height):
        for i in range(fork_height+1, len(chain)-1):
            if chain[i][1] - chain[i+1][1] >= self.difficulty * 240:
                self.log.info("chain %d is heavy at height %d %d %d", chain_id, i,  chain[i][1], chain[i+1][1])
                return
        if chain[-1][1] >= self.difficulty * 240:
            self.log.info("chain %d is heavy at height %d %d %d", chain_id, i,  chain[i][1], chain[i+1][1])

def next_block_time(generation_period):
    time.time() + random.expovariate(1 / generation_period)


class AttackMiningThread(threading.Thread):

    def __init__(self, parent, fork_height, blockset, receipts_root, difficulty, generation_period):
        threading.Thread.__init__(self, daemon=True)
        self.fork_height = fork_height
        self.parent = parent
        self.blockset = blockset
        self.receipts_root = receipts_root
        self.difficulty = difficulty
        self.generation_period = generation_period

    def run(self):
        while True:
            time.sleep(random.expovariate(1 / self.generation_period))
            block = NewBlock(create_block(decode_hex(self.parent), height=self.fork_height+1, deferred_receipts_root=self.receipts_root, difficulty=self.difficulty, timestamp=random.randint(1, 2 ** 31)))
            self.blockset.put(block)

if __name__ == "__main__":
    P2PTest().main()