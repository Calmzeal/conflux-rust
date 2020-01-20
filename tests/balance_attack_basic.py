#!/usr/bin/env python3

##Modules to define class TaskQueue
import collections
import queue
from threading import Thread

##Modules to set up cfx nodes, network environments and rpc calls
from argparse import ArgumentParser
from eth_utils import encode_hex, decode_hex
from conflux.utils import parse_as_int
from test_framework.blocktools import create_block
from test_framework.test_framework import ConfluxTestFramework
from test_framework.mininode import *
from test_framework.util import *
from conflux.rpc import RpcClient


class P2PTest(ConfluxTestFramework):

    def set_test_params(self):
        self.setup_clean_chain = True
        self.num_nodes = 2
        # self.start_attack = True
        self.task_queue = TaskQueue()

        self.total_period = 0.25
        self.latency = 6000
        self.out_degree = 1
        self.termination_time = 5400

        self.env = AttackEnvironment(self.num_nodes)
        self.env.recent_timeout = self.latency
        self.env.debug_allow_borrow = False
        self.env.withold = 0
        self.env.attack_start = False



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

        self.evil_rate = self.options.evil_rate
        self.difficulty = int(self.num_nodes / (1 / self.total_period) / (1 - self.evil_rate) * 100)
        self.env.difficulty = self.difficulty
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
        self.env.nodes = self.nodes

        self.neighbors = []
        for i in range(self.num_nodes):
            peers = set()
            for j in range(self.out_degree):
                peer = random.randint(0, self.num_nodes - 1)
                while peer in peers or peer == i:
                    peer = random.randint(0, self.num_nodes - 1)
                peers.add(peer)
            self.neighbors.append(list(peers))

        print(self.neighbors)
        self.log.info("Connection Started")
        for i in range(self.num_nodes):
            for j in self.neighbors[i]:
                connect_nodes(self.nodes, i, j)
                self.nodes[i].addlatency(self.nodes[j].key, self.latency * random.uniform(0.75, 1.25))

        self.log.info("Connection Finished")


    def run_test(self):
        # Connect mininodes to cfx nodes
        remote = False
        start_p2p_connection(self.nodes, remote, self.task_queue, self.env)

        generation_period = 1 / (1 / self.total_period * self.evil_rate)
        self.log.info("Adversary mining average period=%f", generation_period)

        self.set_attack_env()




        # Executed the attacks
        begin = 0
        count = 0
        scan_freq = 1

        merged = False
        e = self.env
        chain = []

        while True:
            # This roughly simulates adversary's mining power
            if e.attack_start:
                time.sleep(random.expovariate(1 / generation_period))

            # Check test's termination
            count += 1
            if count > 2400 / generation_period:
                self.log.info("Not merged after 40 minutes")
                break

            # Check merge every freq period of attack
            if count%scan_freq == 0:
                chain = self.get_chains()
                hashes = list(map(lambda x:chain[x][e.fork_height][0],range(self.num_nodes)))
                print(hashes)

                #print(values)
                if hashes.count(e.left_fork_hash)==int(self.num_nodes/2) and hashes.count(e.right_fork_hash)==int(self.num_nodes/2) and begin == 0:
                    begin = time.time()
                    e.attack_start = True
                    print("Attack-begins")
                merged = True
                for i in range(self.num_nodes-1):
                    merged &= chain[i][e.fork_height][0] == chain[i+1][e.fork_height][0]

                #self.check_chain_heavy(chain[0], 0, t.fork_height)
                if merged:
                    print(time.time()-begin)
                    self.log.info("Pivot chain merged")
                    self.log.info("Merged. Winner: %s among [%s,%s]", chain[0][e.fork_height][0], e.left_fork_hash, e.right_fork_hash)
                    break

            e.update_subtree_params(e.left_fork_hash, e.right_fork_hash)
            # Decide attack target
            withhold_queue, chirality, target = (e.left_withheld_blocks_queue, e.left_fork_hash, e.nodes_to_keep_left) \
                if e.left_withheld_blocks_queue.qsize() + e.left_subtree_weight \
                   < e.right_withheld_blocks_queue.qsize() + e.right_subtree_weight else \
                (e.right_withheld_blocks_queue, e.right_fork_hash, e.nodes_to_keep_right)

            timestamp=int(time.time())
            withhold_queue.put(NewBlock(
                create_block(decode_hex(chirality), height=e.fork_height + 1, deferred_receipts_root=e.receipts_root,
                difficulty=e.difficulty, timestamp=timestamp,
                author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1)))))
            e.maintain_recent_blocks(timestamp)
            e.adversary_strategy(
                e.left_subtree_weight - e.right_subtree_weight
                - len(e.honest_left_recent_sent_blocks) + len(e.honest_right_recent_sent_blocks),
                timestamp,
                [e.nodes_to_keep_left, e.nodes_to_keep_right],
                [e.left_withheld_blocks_queue, e.right_withheld_blocks_queue],
            )



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

    def scan_chains_successfully(self):
        chain = self.get_chains()
        height = 0
        while True:
            for i in range(self.num_nodes):
                if height >= len(chain[i]):
                    self.log.info("No fork to start attack, retry")
                    return False
            for i in range(self.num_nodes - 1):
                if chain[i][height][0] != chain[i+1][height][0]:
                    lhash = chain[i][height][0]
                    rhash = chain[i + 1][height][0]
                    time.sleep(25)
                    if self.env.update_subtree_params(lhash,rhash):
                        self.log.info("Forked at height %d %s %s", height, chain[i][height], chain[i + 1][height])
                        # Our generated block has height fork_height+1, so its deferred block is fork_height-4
                        receipts_root = decode_hex(self.nodes[i].getExecutedInfo(chain[i][height - 4][0])[0]) \
                            if height >= 5 else default_config["GENESIS_RECEIPTS_ROOT"]
                        self.env.set_params(i, lhash, rhash, height, receipts_root)
                        return True
                    else:
                        self.log.info("Fork block unreceived... retry")
                        return False
            height += 1

    def set_attack_env(self):
        while not self.scan_chains_successfully():
            time.sleep(0.1)



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


class AttackEnvironment:
    """ Define the scale of the network, the attack strategy and parameters
    """
    def __init__(self,num_nodes):
        self.nodes_to_keep_left = list(range(0, num_nodes, 2))
        self.nodes_to_keep_right = list(range(1, num_nodes, 2))

        self.parent_map = {}
        self.weight = {}


        # Initialize adversary.
        self.left_withheld_blocks_queue = queue.Queue()
        self.right_withheld_blocks_queue = queue.Queue()
        self.total_borrowed_blocks = 0
        self.left_borrowed_blocks = 0
        self.right_borrowed_blocks = 0
        self.withhold_done = False
        # The number of recent blocks mined under left side sent to the network.
        self.adv_left_recent_sent_blocks = collections.deque()
        self.adv_right_recent_sent_blocks = collections.deque()
        self.honest_left_recent_sent_blocks = collections.deque()
        self.honest_right_recent_sent_blocks = collections.deque()

        def _lambda(env, node, new_block_hashes):

            for h in new_block_hashes:

                x = RpcClient(node).block_by_hash(encode_hex(h))
                difficulty = int(x['difficulty'], 0)
                if difficulty > env.difficulty * 240:
                    print(f"Heavy block comes from node {node.index} with difficulty {difficulty}")
                relative_weight = int(difficulty/env.difficulty)

                if env.parent_map.get(x["hash"]) is None:
                    original_weight = env.weight.get(x["hash"])

                    if original_weight is None:
                        env.weight.update({x["hash"]: relative_weight})
                        original_weight = relative_weight

                    env.parent_map.update({x["hash"]: x["parentHash"]})
                    child = x["hash"]
                    parent = env.parent_map.get(child)
                    parent_weight = env.weight.get(parent)

                    if parent_weight is None:
                        env.weight.update({parent: original_weight})
                    else:
                        while not (parent is None):
                            parent_weight += original_weight
                            env.weight.update({parent: parent_weight})
                            child = parent
                            parent = env.parent_map.get(child)
                            parent_weight = env.weight.get(parent)

                if hasattr(env,"left_subtree_weight"):
                    env.update_subtree_params(env.left_fork_hash,env.right_fork_hash)
                    env.adversary_strategy(
                        env.left_subtree_weight - env.right_subtree_weight
                        - len(env.honest_left_recent_sent_blocks) + len(env.honest_right_recent_sent_blocks),
                        int(time.time()),
                        [env.nodes_to_keep_left, env.nodes_to_keep_right],
                        [env.left_withheld_blocks_queue, env.right_withheld_blocks_queue],
                    )


        self.parse_block_hashes = _lambda

    def set_params(self,branch_leader,lhash,rhash,fork_height,receipts_root):
        self.left_leader = branch_leader
        self.right_leader = branch_leader + 1
        self.left_fork_hash = lhash
        self.right_fork_hash = rhash
        self.fork_height = fork_height
        self.receipts_root = receipts_root

        if self.left_leader in self.nodes_to_keep_right:
            self.nodes_to_keep_right.remove(self.left_leader)
            self.nodes_to_keep_left.append(self.left_leader)
        if self.right_leader in self.nodes_to_keep_left:
            self.nodes_to_keep_left.remove(self.right_leader)
            self.nodes_to_keep_right.append(self.right_leader)



    def update_subtree_params(self,lhash,rhash):
        # return update being successful or not
        l = self.weight.get(lhash)
        r = self.weight.get(rhash)
        if (l is None) or (r is None):
            return False
        else:
            self.left_subtree_weight = l
            self.right_subtree_weight = r
            return True

    def maintain_recent_blocks(self, timestamp):
        non_recent_timestamp = timestamp - self.recent_timeout
        for recent_sent_blocks in [
            self.adv_left_recent_sent_blocks, self.adv_right_recent_sent_blocks,
            self.honest_left_recent_sent_blocks, self.honest_right_recent_sent_blocks,
        ]:
            while len(recent_sent_blocks) > 0 \
                and recent_sent_blocks[0][0] <= non_recent_timestamp:
                recent_sent_blocks.popleft()

    def adversary_send_withheld_block(self, chirality, target, timestamp):
        if chirality == "L":
            withheld_queue = self.left_withheld_blocks_queue
            recent_sent_blocks = self.adv_left_recent_sent_blocks
        else:
            withheld_queue = self.right_withheld_blocks_queue
            recent_sent_blocks = self.adv_right_recent_sent_blocks
        if withheld_queue.empty():
            blk_map = lambda hash:NewBlock(
                             create_block(decode_hex(hash), height=t.fork_height + 1, deferred_receipts_root=self.receipts_root,
                             difficulty=self.difficulty, timestamp=timestamp,
                             author=decode_hex("%040x" % random.randint(0, 2 ** 32 - 1))))
            if self.debug_allow_borrow:
                self.total_borrowed_blocks += 1
                if chirality == "L":
                    self.left_borrowed_blocks += 1
                    blk = blk_map(self.left_fork_hash)
                else:
                    self.right_borrowed_blocks += 1
                    blk = blk_map(self.right_fork_hash)
            else:
                return
        else:
            blk = withheld_queue.get()

        for i in target:
            self.nodes[i].p2p.send_protocol_msg(blk)
        print("send to %s group block %s", chirality, blk.block.hash_hex())
        self.update_subtree_params(self.left_fork_hash,self.right_fork_hash)
        print(target,self.left_subtree_weight,self.right_subtree_weight)
        recent_sent_blocks.append((timestamp, blk))
        self.maintain_recent_blocks(timestamp)



    def adversary_strategy(self,
                           global_subtree_weight_diff,
                           timestamp,
                           targets,
                           withhold_queues,
                          ):
        if withhold_queues[0].qsize() + withhold_queues[1].qsize() >= self.withold:
            self.withhold_done = True

        adv_recent_sent_left = len(self.adv_left_recent_sent_blocks)
        adv_recent_sent_right = len(self.adv_right_recent_sent_blocks)
        approx_right_target_subtree_weight_diff = global_subtree_weight_diff - adv_recent_sent_left
        approx_left_target_subtree_weight_diff = global_subtree_weight_diff + adv_recent_sent_right
        #extra_send = 0
        left_send_count = -approx_left_target_subtree_weight_diff # + extra_send
        right_send_count = approx_right_target_subtree_weight_diff + 1 # + extra_send

        # Debug output only, estimation.
        """
        honest_recent_mined_left = len(self.honest_left_recent_sent_blocks)
        honest_recent_mined_right = len(self.honest_right_recent_sent_blocks)
        all_received_left = self.left_subtree_weight - adv_recent_sent_left - honest_recent_mined_left
        all_received_right = self.right_subtree_weight - adv_recent_sent_right - honest_recent_mined_right
        left_target_received_left = self.left_subtree_weight - honest_recent_mined_left
        right_target_received_right = self.right_subtree_weight - honest_recent_mined_right

        print(f"Global view before action: ({self.left_subtree_weight}, {self.right_subtree_weight}); "
              f"Honest recent mined: ({honest_recent_mined_left}, {honest_recent_mined_right}), "
              f"Adv recent sent: ({adv_recent_sent_left}, {adv_recent_sent_right}), "
              f"Est. all received: ({all_received_left}, {all_received_right}), "
              f"left received: ({left_target_received_left}, {all_received_right}), "
              f"right received: ({all_received_left}, {right_target_received_right}); "
              f"Adv to send: ({left_send_count}, {right_send_count}) in which extra_send {extra_send}, "
              f"adv withhold: ({self.left_withheld_blocks_queue.qsize()}, {self.right_withheld_blocks_queue.qsize()}); "
              f"adv borrowed blocks: ({self.left_borrowed_blocks}, {self.right_borrowed_blocks})."
        )
        """

        if (self.debug_allow_borrow or self.withhold_done) and left_send_count > 0:
            for i in range(left_send_count):
                self.adversary_send_withheld_block("L",  targets[0], timestamp)

        if (self.debug_allow_borrow or self.withhold_done) and right_send_count > 0:
            for i in range(right_send_count):
                self.adversary_send_withheld_block("R", targets[1], timestamp)


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
