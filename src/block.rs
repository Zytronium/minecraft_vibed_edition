/// Different types of blocks in the world
#[derive(Clone, Copy, PartialEq)]
pub enum BlockType {
    Air,     // Empty space (not rendered)
    Grass,   // Grass block with different textures per face
    Dirt,    // Dirt block
    Stone,   // Stone block
    Log,     // Tree log with bark on sides, rings on top/bottom
    Leaves,  // Tree leaves - semi-transparent foliage
}

impl BlockType {
    /// Returns the texture index for a given face of this block type
    /// Face indices: 0=Left, 1=Right, 2=Bottom, 3=Top, 4=Back, 5=Front
    pub fn get_texture_indices(&self, face: usize) -> u32 {
        match self {
            BlockType::Air => 0,
            BlockType::Grass => {
                match face {
                    3 => 0, // Top face uses grass_top.png
                    2 => 1, // Bottom face uses grass_bottom.png (dirt)
                    _ => 2, // Side faces use grass_side.png
                }
            }
            BlockType::Dirt => 3,   // All faces use dirt.png
            BlockType::Stone => 4,  // All faces use stone.png
            BlockType::Log => {
                match face {
                    2 | 3 => 5, // Top and bottom faces use log_top-bottom.png (rings)
                    _ => 6,     // Side faces use log_side.png (bark)
                }
            }
            BlockType::Leaves => 7, // All faces use leaves.png
        }
    }

    /// Check if this block type is transparent (allows seeing through)
    pub fn is_transparent(&self) -> bool {
        matches!(self, BlockType::Air | BlockType::Leaves)
    }
}
