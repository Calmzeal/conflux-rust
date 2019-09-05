use crate::{
    block_data_manager::{
        BlockExecutionResultWithEpoch, CheckpointHashes,
        ConsensusGraphExecutionInfo, LocalBlockInfo,
    },
    db::{
        COL_BLOCKS, COL_DELTA_TRIE, COL_EPOCH_NUMBER, COL_MISC, COL_TX_ADDRESS,
    },
    verification::VerificationConfig,
};
use byteorder::{ByteOrder, LittleEndian};
use cfx_types::H256;
use db::SystemDB;
use elastic_array::ElasticArray128;
use kvdb::DBTransaction;
use primitives::{Block, BlockHeader, SignedTransaction, TransactionAddress};
use rlp::{Decodable, Encodable, Rlp};
use std::sync::Arc;

const LOCAL_BLOCK_INFO_SUFFIX_BYTE: u8 = 1;
const BLOCK_BODY_SUFFIX_BYTE: u8 = 2;
const BLOCK_EXECUTION_RESULT_SUFFIX_BYTE: u8 = 3;
const EPOCH_EXECUTION_CONTEXT_SUFFIX_BYTE: u8 = 4;
const EPOCH_CONSENSUS_EXECUTION_INFO_SUFFIX_BYTE: u8 = 5;

#[derive(Clone, Copy)]
enum DBTable {
    Misc,
    Blocks,
    Transactions,
    EpochNumbers,
}

fn rocks_db_col(table: DBTable) -> Option<u32> {
    match table {
        DBTable::Misc => COL_MISC,
        DBTable::Blocks => COL_BLOCKS,
        DBTable::Transactions => COL_TX_ADDRESS,
        DBTable::EpochNumbers => COL_EPOCH_NUMBER,
    }
}

pub struct DBManager {
    pub db: Arc<SystemDB>,
}

impl DBManager {
    pub fn block_from_db(&self, block_hash: &H256) -> Option<Block> {
        Some(Block::new(
            self.block_header_from_db(block_hash)?,
            self.block_body_from_db(block_hash)?,
        ))
    }

    pub fn insert_block_header_to_db(&self, header: &BlockHeader) {
        self.insert_encodable_val(DBTable::Blocks, &header.hash(), header);
    }

    pub fn block_header_from_db(&self, hash: &H256) -> Option<BlockHeader> {
        let mut block_header =
            self.load_decodable_val(DBTable::Blocks, hash)?;
        VerificationConfig::compute_header_pow_quality(&mut block_header);
        Some(block_header)
    }

    pub fn remove_block_header_from_db(&self, hash: &H256) {
        self.remove_from_db(DBTable::Blocks, hash);
    }

    pub fn insert_transaction_address_to_db(
        &self, hash: &H256, value: &TransactionAddress,
    ) {
        self.insert_encodable_val(DBTable::Transactions, hash, value)
    }

    pub fn transaction_address_from_db(
        &self, hash: &H256,
    ) -> Option<TransactionAddress> {
        self.load_decodable_val(DBTable::Transactions, hash)
    }

    /// Store block info to db. Block info includes block status and
    /// the sequence number when the block enters consensus graph.
    /// The db key is the block hash plus one extra byte, so we can get better
    /// data locality if we get both a block and its info from db.
    /// The info is not a part of the block because the block is inserted
    /// before we know its info, and we do not want to insert a large chunk
    /// again. TODO Maybe we can use in-place modification (operator `merge`
    /// in rocksdb) to keep the info together with the block.
    pub fn insert_local_block_info_to_db(
        &self, block_hash: &H256, value: &LocalBlockInfo,
    ) {
        self.insert_encodable_val(
            DBTable::Blocks,
            &local_block_info_key(block_hash),
            value,
        );
    }

    /// Get block info from db.
    pub fn local_block_info_from_db(
        &self, block_hash: &H256,
    ) -> Option<LocalBlockInfo> {
        self.load_decodable_val(
            DBTable::Blocks,
            &local_block_info_key(block_hash),
        )
    }

    pub fn insert_block_body_to_db(&self, block: &Block) {
        self.insert_to_db(
            DBTable::Blocks,
            &block_body_key(&block.hash()),
            block.encode_body_with_tx_public(),
        )
    }

    pub fn block_body_from_db(
        &self, hash: &H256,
    ) -> Option<Vec<Arc<SignedTransaction>>> {
        let encoded =
            self.load_from_db(DBTable::Blocks, &block_body_key(hash))?;
        let rlp = Rlp::new(&encoded);
        Some(
            Block::decode_body_with_tx_public(&rlp)
                .expect("Wrong block rlp format!"),
        )
    }

    pub fn remove_block_body_from_db(&self, hash: &H256) {
        self.remove_from_db(DBTable::Blocks, &block_body_key(hash))
    }

    pub fn insert_block_execution_result_to_db(
        &self, hash: &H256, value: &BlockExecutionResultWithEpoch,
    ) {
        self.insert_encodable_val(
            DBTable::Blocks,
            &block_execution_result_key(hash),
            value,
        )
    }

    pub fn block_execution_result_from_db(
        &self, hash: &H256,
    ) -> Option<BlockExecutionResultWithEpoch> {
        self.load_decodable_val(
            DBTable::Blocks,
            &block_execution_result_key(hash),
        )
    }

    pub fn insert_checkpoint_hashes_to_db(
        &self, checkpoint_prev: &H256, checkpoint_cur: &H256,
    ) {
        self.insert_encodable_val(
            DBTable::Misc,
            b"checkpoint",
            &CheckpointHashes::new(*checkpoint_prev, *checkpoint_cur),
        );
    }

    pub fn checkpoint_hashes_from_db(&self) -> Option<(H256, H256)> {
        let checkpoints: CheckpointHashes =
            self.load_decodable_val(DBTable::Misc, b"checkpoint")?;
        Some((checkpoints.prev_hash, checkpoints.cur_hash))
    }

    pub fn insert_epoch_set_hashes_to_db(
        &self, epoch: u64, hashes: &Vec<H256>,
    ) {
        self.insert_encodable_list(
            DBTable::EpochNumbers,
            &epoch_set_key(epoch)[0..8],
            hashes,
        );
    }

    pub fn epoch_set_hashes_from_db(&self, epoch: u64) -> Option<Vec<H256>> {
        self.load_decodable_list(
            DBTable::EpochNumbers,
            &epoch_set_key(epoch)[0..8],
        )
    }

    pub fn insert_terminals_to_db(&self, terminals: &Vec<H256>) {
        self.insert_encodable_list(DBTable::Misc, b"terminals", terminals);
    }

    pub fn terminals_from_db(&self) -> Option<Vec<H256>> {
        self.load_decodable_list(DBTable::Misc, b"terminals")
    }

    pub fn insert_consensus_graph_execution_info_to_db(
        &self, hash: &H256, ctx: &ConsensusGraphExecutionInfo,
    ) {
        self.insert_encodable_val(
            DBTable::Blocks,
            &epoch_consensus_execution_info_key(hash),
            ctx,
        );
    }

    pub fn consensus_graph_execution_info_from_db(
        &self, hash: &H256,
    ) -> Option<ConsensusGraphExecutionInfo> {
        self.load_decodable_val(
            DBTable::Blocks,
            &epoch_consensus_execution_info_key(hash),
        )
    }

    pub fn insert_instance_id_to_db(&self, instance_id: u64) {
        self.insert_encodable_val(DBTable::Misc, b"instance", &instance_id);
    }

    pub fn instance_id_from_db(&self) -> Option<u64> {
        self.load_decodable_val(DBTable::Misc, b"instance")
    }

    /// The functions below are private utils used by the DBManager to access
    /// database
    fn insert_to_db(&self, table: DBTable, db_key: &[u8], value: Vec<u8>) {
        let mut dbops = self.db.key_value().transaction();
        dbops.put(rocks_db_col(table), db_key, &value);
        self.commit_db_transaction(dbops);
    }

    fn remove_from_db(&self, table: DBTable, db_key: &[u8]) {
        let mut dbops = self.db.key_value().transaction();
        dbops.delete(rocks_db_col(table), db_key);
        self.commit_db_transaction(dbops);
    }

    fn commit_db_transaction(&self, transaction: DBTransaction) {
        self.db
            .key_value()
            .write(transaction)
            .expect("crash for db failure");
    }

    fn load_from_db(
        &self, table: DBTable, db_key: &[u8],
    ) -> Option<ElasticArray128<u8>> {
        self.db.key_value().get(rocks_db_col(table), db_key).expect("Low level database error when fetching transaction index. Some issue with disk?")
    }

    fn insert_encodable_val<V>(
        &self, table: DBTable, db_key: &[u8], value: &V,
    ) where V: Encodable {
        self.insert_to_db(table, db_key, rlp::encode(value))
    }

    fn insert_encodable_list<V>(
        &self, table: DBTable, db_key: &[u8], value: &Vec<V>,
    ) where V: Encodable {
        self.insert_to_db(table, db_key, rlp::encode_list(value))
    }

    fn load_decodable_val<V>(
        &self, table: DBTable, db_key: &[u8],
    ) -> Option<V>
    where V: Decodable {
        let encoded = self.load_from_db(table, db_key)?;
        Some(Rlp::new(&encoded).as_val().expect("decode succeeds"))
    }

    fn load_decodable_list<V>(
        &self, table: DBTable, db_key: &[u8],
    ) -> Option<Vec<V>>
    where V: Decodable {
        let encoded = self.load_from_db(table, db_key)?;
        Some(Rlp::new(&encoded).as_list().expect("decode succeeds"))
    }
}

fn append_suffix(h: &H256, suffix: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(h.len() + 1);
    key.extend_from_slice(&h);
    key.push(suffix);
    key
}

fn local_block_info_key(block_hash: &H256) -> Vec<u8> {
    append_suffix(block_hash, LOCAL_BLOCK_INFO_SUFFIX_BYTE)
}

fn block_body_key(block_hash: &H256) -> Vec<u8> {
    append_suffix(block_hash, BLOCK_BODY_SUFFIX_BYTE)
}

fn epoch_set_key(epoch_number: u64) -> [u8; 8] {
    let mut epoch_key = [0; 8];
    LittleEndian::write_u64(&mut epoch_key[0..8], epoch_number);
    epoch_key
}

fn block_execution_result_key(hash: &H256) -> Vec<u8> {
    append_suffix(hash, BLOCK_EXECUTION_RESULT_SUFFIX_BYTE)
}

fn epoch_execution_context_key(hash: &H256) -> Vec<u8> {
    append_suffix(hash, EPOCH_EXECUTION_CONTEXT_SUFFIX_BYTE)
}

fn epoch_consensus_execution_info_key(hash: &H256) -> Vec<u8> {
    append_suffix(hash, EPOCH_CONSENSUS_EXECUTION_INFO_SUFFIX_BYTE)
}
