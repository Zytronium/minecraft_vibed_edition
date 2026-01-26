use crate::models;
use std::collections::HashMap;

/// Block ID type - now dynamic instead of hardcoded enum
pub type BlockId = u16;

/// Special block IDs that are always present
pub const AIR_ID: BlockId = 0;

/// Runtime block type that references the registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockType {
    pub id: BlockId,
}

impl BlockType {
    pub const fn new(id: BlockId) -> Self {
        Self { id }
    }

    pub const fn air() -> Self {
        Self::new(AIR_ID)
    }
}

/// Block registry that maps IDs to block definitions
pub struct BlockRegistry {
    blocks: Vec<models::Block>,
    id_map: HashMap<String, BlockId>,
    texture_indices: HashMap<BlockId, [u32; 6]>, // 6 faces per block
}

impl BlockRegistry {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            id_map: HashMap::new(),
            texture_indices: HashMap::new(),
        }
    }

    /// Load blocks from JSON data and assign IDs
    pub fn load_blocks(&mut self, blocks: Vec<models::Block>) {
        // Reserve ID 0 for air
        self.id_map.insert("air".to_string(), AIR_ID);

        let mut next_id = 1u16;
        for block in blocks {
            self.id_map.insert(block.id.clone(), next_id);
            self.blocks.push(block);
            next_id += 1;
        }
    }

    /// Build texture index mapping after textures are loaded
    /// texture_map: maps texture names -> texture array indices
    pub fn build_texture_indices(&mut self, texture_map: &HashMap<String, u32>) {
        for block in &self.blocks {
            if let Some(&block_id) = self.id_map.get(&block.id) {
                let mut indices = [0u32; 6];

                // Get texture index for each face
                // Face order: 0=West, 1=East, 2=Bottom, 3=Top, 4=North, 5=South
                for face in 0..6 {
                    let texture_name = block.textures.get_texture(face);
                    indices[face] = *texture_map.get(texture_name).unwrap_or(&0);
                }

                self.texture_indices.insert(block_id, indices);
            }
        }
    }

    /// Get block ID by string identifier
    pub fn get_id(&self, block_name: &str) -> Option<BlockId> {
        self.id_map.get(block_name).copied()
    }

    /// Get block definition by ID
    pub fn get_block(&self, id: BlockId) -> Option<&models::Block> {
        if id == AIR_ID {
            return None; // Air has no block data
        }
        self.blocks.get((id - 1) as usize)
    }

    /// Check if block is transparent
    pub fn is_transparent(&self, id: BlockId) -> bool {
        if id == AIR_ID {
            return true;
        }
        self.get_block(id).map(|b| b.transparent).unwrap_or(false)
    }

    /// Get texture indices for a specific face of a block
    pub fn get_texture_indices(&self, id: BlockId, face: usize) -> u32 {
        self.texture_indices
            .get(&id)
            .and_then(|indices| indices.get(face))
            .copied()
            .unwrap_or(0)
    }

    /// Create a BlockType from a string ID
    pub fn block_type(&self, block_name: &str) -> BlockType {
        BlockType::new(self.get_id(block_name).unwrap_or(AIR_ID))
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}
