#!/usr/bin/env python3
from http.client import CannotSendRequest
from eth_utils import decode_hex
from conflux.utils import encode_hex, privtoaddr, parse_as_int
from test_framework.blocktools import create_transaction, encode_hex_0x
from test_framework.test_framework import ConfluxTestFramework
from test_framework.mininode import *
from test_framework.util import *
from web3 import Web3
from easysolc import Solc

class P2PTest(ConfluxTestFramework):
    def set_test_params(self):
        self.setup_clean_chain = True
        self.num_nodes = 8

    def setup_network(self):
        self.setup_nodes()
        connect_sample_nodes(self.nodes, self.log)
        sync_blocks(self.nodes)

    def run_test(self):
        # Prevent easysolc from configuring the root logger to print to stderr
        self.log.propagate = False

        solc = Solc()
        erc20_contract = solc.get_contract_instance(source=os.path.dirname(os.path.realpath(__file__)) + "/erc20.sol", contract_name="FixedSupplyToken")

        start_p2p_connection(self.nodes)

        self.log.info("Initializing contract")
        genesis_key = default_config["GENESIS_PRI_KEY"]
        genesis_addr = privtoaddr(genesis_key)
        nonce = 0
        gas_price = 1
        gas = 50000000
        block_gen_thread = BlockGenThread(self.nodes, self.log)
        block_gen_thread.start()
        self.tx_conf = {"from":Web3.toChecksumAddress(encode_hex_0x(genesis_addr)), "nonce":int_to_hex(nonce), "gas":int_to_hex(gas), "gasPrice":int_to_hex(gas_price)}
        raw_create = erc20_contract.constructor().buildTransaction(self.tx_conf)
        tx_data = decode_hex(raw_create["data"])
        tx_create = create_transaction(pri_key=genesis_key, receiver=b'', nonce=nonce, gas_price=gas_price, data=tx_data, gas=gas, value=0)
        self.nodes[0].p2p.send_protocol_msg(Transactions(transactions=[tx_create]))
        self.wait_for_tx([tx_create])
        self.log.info("Contract created, start transfering tokens")

    def get_balance(self, contract, token_address, nonce):
        tx = contract.functions.balanceOf(Web3.toChecksumAddress(encode_hex(token_address))).buildTransaction(self.tx_conf)
        tx["value"] = int_to_hex(tx['value'])
        tx["hash"] = "0x"+"0"*64
        tx["nonce"] = int_to_hex(nonce)
        tx["v"] = "0x0"
        tx["r"] = "0x0"
        tx["s"] = "0x0"
        result = self.nodes[0].cfx_call(tx)
        balance = bytes_to_int(decode_hex(result))
        self.log.debug("address=%s, balance=%s", encode_hex(token_address), balance)
        return balance


if __name__ == "__main__":
    P2PTest().main()