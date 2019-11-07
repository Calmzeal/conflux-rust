1. Format of blocks received by mininodes
0
{
'adaptive'  :   False,
'blame'     :   0,
'deferredLogsBloomHash' :   '0xd397b3b043d87fcd6fad1291ff0bfd16401c274896d8c63a923727f077b8e0b5',
'deferredReceiptsRoot'  :   '0x959684cc863003d5ac5cb31bcf5baf7e1b4fc60963fcc36fbc1bf4394a0e2e3c',
'deferredStateRoot'     :   '0x0c39492aa99d81ef66ac181639b786c029c497c24c087d28fcfe270e0931b6a0',
'deferredStateRootWithAux': {
    'auxInfo': {
        'intermediateDeltaEpochId'  :   '0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470',
        'previousSnapshotRoot'      :   '0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470'
    },
    'stateRoot': {
        'deltaRoot'             :   '0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470',
        'intermediateDeltaRoot' :   '0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470',
        'snapshotRoot'          :   '0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470'
    }
},
'difficulty'    :   '0x3e',
'epochNumber'   :   None,
'gasLimit'      :   '0xb2d05e00',
'hash'          :   '0x5ee76a54f0f694fece79ddb1de3b39c226ec1957aaef5295348b96d58cdf685b',
'height'        :   '0x9',
'miner'         :   '0x000000000000000000000000000000005bc00ad7',
'nonce'         :   '0xd4',
'parentHash'    :   '0x665f9d39ee5f822bca1a7c0444cedff2613c5d7c854d25624f1ce836aa663d47',
'refereeHashes' :   [],
'size'          :   '0x0',
'stable'        :   True,
'timestamp'     :   '0x5db7373b',
'transactions'  :   [],
'transactionsRoot'  :   '0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347'
}

2. Format of the genesis block of conflux node
Block {
block_header:
    BlockHeader {
        rlp_part : BlockHeaderRlpPart {
            parent_hash : 0x0000000000000000000000000000000000000000000000000000000000000000,
            height      : 0,
            timestamp   : 0,
            author      : 0x000000000000000000000000000000000000000a,
            transactions_root       :   0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347,
            deferred_state_root     :   0x0c39492aa99d81ef66ac181639b786c029c497c24c087d28fcfe270e0931b6a0,
            deferred_receipts_root  :   0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347,
            deferred_logs_bloom_hash:   0xd397b3b043d87fcd6fad1291ff0bfd16401c274896d8c63a923727f077b8e0b5,
            blame       : 0,
            difficulty  : 0,
            adaptive    : false,
            gas_limit   : 3000000000,
            referee_hashes: [],
            nonce       : 0
        },
        hash        :   Some(0xdd5cefe8c8f0a89c5c3c296cde0d9af39f561c1d2afd2314a93c87c7fecfda8f),
        pow_quality :   0,
        approximated_rlp_size : 304,
        state_root_with_aux_info : StateRootWithAuxInfo {
            state_root : StateRoot {
                snapshot_root           :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470,
                intermediate_delta_root :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470,
                delta_root              :   0x5a79b0adf83ff7ea83c7862e825e1ad64af0c4492a4e0f7a48610cb130fbfb14
            },
            aux_info : StateRootAuxInfo {
                previous_snapshot_root      :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470,
                intermediate_delta_epoch_id :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
            }
        }
    },
    transactions            : [],
    approximated_rlp_size   : 304,
    approximated_rlp_size_with_public : 304
}

cfxcore::consensus::consensus_inner - Block 0xdd5c…da8f inserted into Consensus with index=0 past_weight=0

client::archive - Start mining with pow config:
ProofOfWorkConfig {
    test_mode   :   true,
    initial_difficulty  :   62,
    block_generation_period     :   1000000,
    difficulty_adjustment_epoch_period  :   20
}
cfxcore::transaction_pool::transaction_pool_inner - After packing packed_transactions: 0, rlp size: 0
cfxcore::consensus::consensus_inner - total_weight before insert: 0
cfxcore::consensus::consensus_inner - block is stable: too close to genesis, adjusted beta 186000

blockgen - mined block
Get solution 16839052365394136053

cfxcore::sync::synchronization_protocol_handler - Mined block 0xc74a653a35117a3afd1726777f36948fed104abed7e9fc9515df1e899e07fd66
header = BlockHeader {
    rlp_part : BlockHeaderRlpPart {
        parent_hash :   0xdd5cefe8c8f0a89c5c3c296cde0d9af39f561c1d2afd2314a93c87c7fecfda8f,
        height      :   1,
        timestamp   :   1572284123,
        author      :   0x0000000000000000000000000000000000000000,
        transactions_root       :   0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347,
        deferred_state_root     :   0x0c39492aa99d81ef66ac181639b786c029c497c24c087d28fcfe270e0931b6a0,
        deferred_receipts_root  :   0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347,
        deferred_logs_bloom_hash:   0xd397b3b043d87fcd6fad1291ff0bfd16401c274896d8c63a923727f077b8e0b5,
        blame       :   0,
        difficulty  :   62,
        adaptive    :   false,
        gas_limit   :   3000000000,
        referee_hashes : [],
        nonce       :   16839052365394136053
    },
    hash        :   Some(0xc74a653a35117a3afd1726777f36948fed104abed7e9fc9515df1e899e07fd66),
    pow_quality :   0,
    approximated_rlp_size : 304,
    state_root_with_aux_info : StateRootWithAuxInfo {
        state_root : StateRoot {
            snapshot_root           :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470,
            intermediate_delta_root :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470,
            delta_root              :   0x5a79b0adf83ff7ea83c7862e825e1ad64af0c4492a4e0f7a48610cb130fbfb14
        },
        aux_info : StateRootAuxInfo {
            previous_snapshot_root      :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470,
            intermediate_delta_epoch_id :   0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
        }
    }
}

 client::rpc::impls::common - RPC Request :
 add_peer(
    NodeEntry {
        id: 0x345672c042cb5f9bbef74f58ee415e888ed3631e0294bc66926eb5dc3a099a16391574c923053cedf8b7ff183cc161535994b70238b1215a0f4d4a8b02db5520,
        endpoint : NodeEndpoint {
            address : V4(127.0.0.1:15691),
            udp_port: 15691
        }
    }
 )

cfxcore::consensus::consensus_inner::consensus_new_block_handler - Block 0x8122…d8b6 anticone size 0

network::service - new session created,
token = 0,
address = V4(127.0.0.1:15691),
id = Some(0x345672c042cb5f9bbef74f58ee415e888ed3631e0294bc66926eb5dc3a099a16391574c923053cedf8b7ff183cc161535994b70238b1215a0f4d4a8b02db5520)


cfxcore::message - Send message(NewBlockHashes) to 0x345672c042cb5f9bbef74f58ee415e888ed3631e0294bc66926eb5dc3a099a16391574c923053cedf8b7ff183cc161535994b70238b1215a0f4d4a8b02db5520

cfxcore::statistics - Statistics: StatisticsInner { sync_graph: SyncGraphStatistics { inserted_block_count: 14 }, consensus_graph: ConsensusGraphStatistics { inserted_block_count: 14, processed_block_count: 13 } }

