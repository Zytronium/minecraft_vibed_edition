use std::collections::HashMap;
use std::sync::Arc;
use noise::{NoiseFn, Perlin};
use crate::block::{BlockType, BlockRegistry};
use crate::vertex::{Vertex, get_face_vertices};

// Chunk dimensions - standard Minecraft sizes
pub const CHUNK_WIDTH: usize = 16;   // Horizontal size (X and Z)
pub const CHUNK_HEIGHT: usize = 128; // Vertical size (Y) - like Minecraft 1.18+

/// A 16x128x16 section of the world
pub struct Chunk {
    blocks: [[[BlockType; CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
    offset_x: i32,  // Chunk position in world (X * CHUNK_WIDTH)
    offset_z: i32,  // Chunk position in world (Z * CHUNK_WIDTH)
}

/// World containing all chunks with cross-chunk block access
pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
    pub registry: Arc<BlockRegistry>,
}

/// Per-chunk mesh data to avoid creating massive monolithic buffers
#[derive(Debug)]
pub struct ChunkMeshData {
    pub opaque_vertices: Vec<Vertex>,
    pub opaque_indices: Vec<u32>,
    pub transparent_vertices: Vec<Vertex>,
    pub transparent_indices: Vec<u32>,
    pub chunk_x: i32,
    pub chunk_z: i32,
}

/// Smooth interpolation function (smoothstep) for blending values
/// Returns 0 when x <= edge0, 1 when x >= edge1, smooth curve in between
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl World {
    pub fn new(registry: Arc<BlockRegistry>) -> Self {
        Self {
            chunks: HashMap::new(),
            registry,
        }
    }

    /// Get block at world coordinates, works across chunk boundaries
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return BlockType::air();
        }

        // Convert world coordinates to chunk coordinates
        let chunk_x = x.div_euclid(CHUNK_WIDTH as i32);
        let chunk_z = z.div_euclid(CHUNK_WIDTH as i32);
        let local_x = x.rem_euclid(CHUNK_WIDTH as i32);
        let local_z = z.rem_euclid(CHUNK_WIDTH as i32);

        if let Some(chunk) = self.chunks.get(&(chunk_x, chunk_z)) {
            chunk.blocks[local_x as usize][y as usize][local_z as usize]
        } else {
            BlockType::air()
        }
    }

    /// Set block at world coordinates, works across chunk boundaries
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return;
        }

        let chunk_x = x.div_euclid(CHUNK_WIDTH as i32);
        let chunk_z = z.div_euclid(CHUNK_WIDTH as i32);
        let local_x = x.rem_euclid(CHUNK_WIDTH as i32);
        let local_z = z.rem_euclid(CHUNK_WIDTH as i32);

        if let Some(chunk) = self.chunks.get_mut(&(chunk_x, chunk_z)) {
            chunk.blocks[local_x as usize][y as usize][local_z as usize] = block;
        }
    }

    /// Generate entire world with terrain and trees
    /// Returns a Vec of per-chunk mesh data instead of combining everything
    /// progress_callback is called with (current_chunk, total_chunks) for progress updates
    pub fn generate<F>(
        world_size_chunks: i32,
        registry: Arc<BlockRegistry>,
        mut progress_callback: F
    ) -> (Self, Vec<ChunkMeshData>)
    where
        F: FnMut(usize, usize),
    {
        let total_chunks = (world_size_chunks * world_size_chunks) as usize;

        let mut world = World::new(Arc::clone(&registry));
        let tree_perlin = Perlin::new(5555);

        let mut current_chunk = 0;

        // First pass: Generate all chunks
        for chunk_x in 0..world_size_chunks {
            for chunk_z in 0..world_size_chunks {
                let chunk = Chunk::new(chunk_x, chunk_z, &registry);
                world.chunks.insert((chunk_x, chunk_z), chunk);

                current_chunk += 1;
                progress_callback(current_chunk, total_chunks);
            }
        }

        // Block types we'll use (with minecraft: prefix)
        let grass_block = registry.block_type("minecraft:grass");
        let log_block = registry.block_type("minecraft:log");
        let leaves_block = registry.block_type("minecraft:leaves");
        let air_block = BlockType::air();

        // Second pass: Generate trees (can now span chunk boundaries)
        for chunk_x in 0..world_size_chunks {
            for chunk_z in 0..world_size_chunks {
                let world_offset_x = chunk_x * CHUNK_WIDTH as i32;
                let world_offset_z = chunk_z * CHUNK_WIDTH as i32;

                for x in 0..CHUNK_WIDTH {
                    for z in 0..CHUNK_WIDTH {
                        let world_x = world_offset_x + x as i32;
                        let world_z = world_offset_z + z as i32;

                        // Find surface height
                        let mut surface_y = 0;
                        for y in (0..CHUNK_HEIGHT).rev() {
                            if world.get_block(world_x, y as i32, world_z) != air_block {
                                surface_y = y as i32;
                                break;
                            }
                        }

                        let surface_block = world.get_block(world_x, surface_y, world_z);
                        if surface_block == grass_block && surface_y > 50 && surface_y < 100 {
                            let tree_noise = tree_perlin.get([world_x as f64 / 4.0, world_z as f64 / 4.0]);

                            // Determine tree density based on biome
                            let nx = world_x as f64 / 100.0;
                            let nz = world_z as f64 / 100.0;
                            let biome_perlin = Perlin::new(1337);
                            let biome_noise = biome_perlin.get([nx * 0.3, nz * 0.3]);
                            let mountain_weight = smoothstep(0.2, 0.5, biome_noise);
                            let plains_weight = smoothstep(-0.4, 0.0, biome_noise) * (1.0 - mountain_weight);

                            // Made trees much less common (higher threshold = fewer trees)
                            let tree_density = if mountain_weight > 0.5 {
                                0.92  // Very sparse trees in mountains
                            } else if plains_weight > 0.5 {
                                0.88  // Less common in plains
                            } else {
                                0.90  // Less common in other biomes
                            };

                            if tree_noise > tree_density {
                                let tree_height = 5 + ((tree_noise * 100.0) as i32 % 2);
                                generate_tree(&mut world, world_x, surface_y + 1, world_z, tree_height, log_block, leaves_block);
                            }
                        }
                    }
                }
            }
        }

        // Third pass: Generate per-chunk meshes with cross-chunk data
        // IMPORTANT: We now return separate mesh data for each chunk instead of combining
        let mut chunk_meshes = Vec::new();

        for chunk_x in 0..world_size_chunks {
            for chunk_z in 0..world_size_chunks {
                if let Some(chunk) = world.chunks.get(&(chunk_x, chunk_z)) {
                    let (opaque_verts, opaque_idx, trans_verts, trans_idx) = chunk.generate_mesh(&world);

                    chunk_meshes.push(ChunkMeshData {
                        opaque_vertices: opaque_verts,
                        opaque_indices: opaque_idx,
                        transparent_vertices: trans_verts,
                        transparent_indices: trans_idx,
                        chunk_x,
                        chunk_z,
                    });
                }
            }
        }

        (world, chunk_meshes)
    }

    /// Regenerate mesh for a specific chunk after block modifications
    /// Returns new mesh data for that chunk
    pub fn regenerate_chunk_mesh(&self, chunk_x: i32, chunk_z: i32) -> Option<ChunkMeshData> {
        if let Some(chunk) = self.chunks.get(&(chunk_x, chunk_z)) {
            let (opaque_verts, opaque_idx, trans_verts, trans_idx) = chunk.generate_mesh(self);

            Some(ChunkMeshData {
                opaque_vertices: opaque_verts,
                opaque_indices: opaque_idx,
                transparent_vertices: trans_verts,
                transparent_indices: trans_idx,
                chunk_x,
                chunk_z,
            })
        } else {
            None
        }
    }
}

impl Chunk {
    /// Generate a new chunk with procedural terrain at the given chunk coordinates
    fn new(chunk_x: i32, chunk_z: i32, registry: &BlockRegistry) -> Self {
        let mut blocks = [[[BlockType::air(); CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];

        // Get block types we'll use
        let air_block = BlockType::air();
        let grass_block = registry.block_type("minecraft:grass");
        let dirt_block = registry.block_type("minecraft:dirt");
        let stone_block = registry.block_type("minecraft:stone");
        let bedrock_block = registry.block_type("minecraft:bedrock");

        // Initialize multiple Perlin noise generators with different seeds for variety
        let terrain_perlin = Perlin::new(42);      // Primary terrain shape
        let biome_perlin = Perlin::new(1337);      // Biome selection
        let cave_perlin = Perlin::new(9999);       // 3D cave system
        let detail_perlin = Perlin::new(7777);     // Fine surface details

        // World offset for this chunk
        let world_offset_x = chunk_x * CHUNK_WIDTH as i32;
        let world_offset_z = chunk_z * CHUNK_WIDTH as i32;

        // Generate terrain for each column in the chunk
        for x in 0..CHUNK_WIDTH {
            for z in 0..CHUNK_WIDTH {
                // Calculate world coordinates for noise sampling
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Normalize coordinates for noise sampling
                let nx = world_x / 100.0;  // Scale for large features
                let nz = world_z / 100.0;

                // === BIOME SYSTEM ===
                // Use low-frequency noise to create distinct biome regions
                let biome_noise = biome_perlin.get([nx * 0.3, nz * 0.3]);

                // === TERRAIN HEIGHT GENERATION ===
                // Generate height for each biome type independently

                // MOUNTAIN BIOME - Tall, dramatic peaks
                let mountain_base = terrain_perlin.get([nx * 0.8, nz * 0.8]) * 35.0;
                let mountain_peaks = terrain_perlin.get([nx * 1.5, nz * 1.5]).abs() * 25.0;
                let mountain_ridges = terrain_perlin.get([nx * 3.0, nz * 3.0]).abs() * 15.0;
                let mountain_height = 64.0 + mountain_base + mountain_peaks + mountain_ridges;

                // HILLS BIOME - Rolling terrain with moderate elevation
                let hills_base = terrain_perlin.get([nx * 1.2, nz * 1.2]) * 15.0;
                let hills_medium = terrain_perlin.get([nx * 2.5, nz * 2.5]) * 10.0;
                let hills_small = detail_perlin.get([nx * 5.0, nz * 5.0]) * 4.0;
                let hills_height = 64.0 + hills_base + hills_medium + hills_small;

                // PLAINS BIOME - Flat with gentle undulation
                let plains_base = terrain_perlin.get([nx * 2.0, nz * 2.0]) * 3.0;
                let plains_gentle = detail_perlin.get([nx * 4.0, nz * 4.0]) * 2.0;
                let plains_height = 64.0 + plains_base + plains_gentle;

                // VALLEY BIOME - Lower elevation, sometimes below sea level
                let valley_base = terrain_perlin.get([nx * 1.0, nz * 1.0]) * 8.0;
                let valley_depression = terrain_perlin.get([nx * 0.6, nz * 0.6]) * -12.0;
                let valley_details = detail_perlin.get([nx * 3.5, nz * 3.5]) * 3.0;
                let valley_height = 60.0 + valley_base + valley_depression + valley_details;

                // === SMOOTH BIOME BLENDING ===
                // Use smoothstep interpolation for gradual transitions between biomes
                let mountain_weight = smoothstep(0.2, 0.5, biome_noise);
                let hills_weight = smoothstep(-0.1, 0.2, biome_noise) * (1.0 - mountain_weight);
                let plains_weight = smoothstep(-0.4, 0.0, biome_noise) * (1.0 - mountain_weight - hills_weight);
                let valley_weight = 1.0 - mountain_weight - hills_weight - plains_weight;

                // Blend all biome heights together based on weights
                let height = mountain_height * mountain_weight
                    + hills_height * hills_weight
                    + plains_height * plains_weight
                    + valley_height * valley_weight;

                // Add fine surface detail to all biomes
                let surface_detail = detail_perlin.get([world_x / 8.0, world_z / 8.0]) * 1.5;
                let final_height = (height + surface_detail).max(35.0).min(120.0);
                let surface_height = final_height as i32;

                // === FILL COLUMN WITH BLOCKS ===
                for y in 0..CHUNK_HEIGHT {
                    blocks[x][y][z] = if y < 2 {
                        // Bedrock layer at the very bottom (Y 0-1)
                        bedrock_block
                    } else if y == 2 {
                        // Random bedrock/stone layer at Y=2
                        if rand::random::<bool>() {
                            bedrock_block
                        } else {
                            stone_block
                        }
                    } else if y < surface_height as usize {
                        let depth = surface_height as usize - y;

                        // === CAVE GENERATION ===
                        // Use 3D Perlin noise to create complex cave networks
                        let cave_scale = 24.0;
                        let cave_noise1 = cave_perlin.get([
                            world_x / cave_scale,
                            y as f64 / cave_scale,
                            world_z / cave_scale
                        ]);

                        // Second octave for more complex cave shapes
                        let cave_noise2 = cave_perlin.get([
                            world_x / (cave_scale * 0.5) + 500.0,
                            y as f64 / (cave_scale * 0.5),
                            world_z / (cave_scale * 0.5) + 500.0
                        ]);

                        // Combine noise layers - caves where both are high
                        let cave_value = cave_noise1 * 0.6 + cave_noise2 * 0.4;

                        // Create larger caves at lower depths
                        let depth_factor = ((y as f32 - 3.0) / 60.0).clamp(0.0, 1.0);
                        let cave_threshold = 0.55 - (depth_factor * 0.15);

                        // Carve caves, but protect surface and bedrock
                        let is_cave = cave_value > cave_threshold.into() && y > 5 && depth > 4;

                        // === SURFACE CAVES ===
                        // Sometimes caves break through to the surface creating dramatic openings
                        let surface_cave = if depth <= 4 && y > 50 {
                            let surface_cave_noise = cave_perlin.get([world_x / 40.0, world_z / 40.0]);
                            surface_cave_noise > 0.7
                        } else {
                            false
                        };

                        if is_cave || surface_cave {
                            air_block
                        } else {
                            // === SURFACE LAYER COMPOSITION ===
                            if depth == 1 {
                                grass_block  // Top layer is grass
                            } else if depth <= 5 {
                                dirt_block   // Next 2-5 layers are dirt
                            } else {
                                stone_block  // Everything deeper is stone
                            }
                        }
                    } else {
                        air_block  // Above surface - air
                    };
                }
            }
        }

        Self {
            blocks,
            offset_x: world_offset_x,
            offset_z: world_offset_z,
        }
    }

    /// Generate mesh geometry for this chunk with cross-chunk neighbor checking
    /// Returns (opaque_vertices, opaque_indices, transparent_vertices, transparent_indices)
    fn generate_mesh(&self, world: &World) -> (Vec<Vertex>, Vec<u32>, Vec<Vertex>, Vec<u32>) {
        let mut opaque_vertices = Vec::new();
        let mut opaque_indices = Vec::new();
        let mut transparent_vertices = Vec::new();
        let mut transparent_indices = Vec::new();

        let air_block = BlockType::air();
        let registry = &world.registry;

        // Iterate through every block in the chunk
        for x in 0..CHUNK_WIDTH as i32 {
            for y in 0..CHUNK_HEIGHT as i32 {
                for z in 0..CHUNK_WIDTH as i32 {
                    let world_x = self.offset_x + x;
                    let world_z = self.offset_z + z;
                    let block = world.get_block(world_x, y, world_z);

                    // Skip air blocks - nothing to render
                    if block == air_block {
                        continue;
                    }

                    // World position of this block
                    let pos = [world_x as f32, y as f32, world_z as f32];
                    let is_transparent = registry.is_transparent(block.id);

                    // Check all 6 faces of the block
                    // Format: (offset to check, face_id)
                    let faces = [
                        ([-1, 0, 0], 0), // Left face (-X)
                        ([ 1, 0, 0], 1), // Right face (+X)
                        ([0, -1, 0], 2), // Bottom face (-Y)
                        ([0,  1, 0], 3), // Top face (+Y)
                        ([0, 0, -1], 4), // Back face (-Z)
                        ([0, 0,  1], 5), // Front face (+Z)
                    ];

                    for (offset, face_id) in faces {
                        let neighbor = world.get_block(
                            world_x + offset[0],
                            y + offset[1],
                            world_z + offset[2]
                        );

                        // Render face if ANY of these conditions are true:
                        // 1. Neighbor is air (always render against air)
                        // 2. Current block is transparent AND neighbor is different type
                        // 3. Current block is opaque AND neighbor is transparent but not air
                        let should_render = neighbor == air_block ||
                            (is_transparent && neighbor != block) ||
                            (!is_transparent && registry.is_transparent(neighbor.id) && neighbor != air_block);

                        if !should_render {
                            continue;
                        }

                        // Choose which vertex/index lists to use based on block transparency
                        let (vertices, indices) = if is_transparent {
                            (&mut transparent_vertices, &mut transparent_indices)
                        } else {
                            (&mut opaque_vertices, &mut opaque_indices)
                        };

                        // Generate the 4 vertices for this face
                        let base_index = vertices.len() as u32;
                        let texture_index = registry.get_texture_indices(block.id, face_id);
                        let face_verts = get_face_vertices(pos, face_id, texture_index);
                        vertices.extend_from_slice(&face_verts);

                        // Create two triangles from the 4 vertices (quad)
                        // Triangle 1: 0-1-2, Triangle 2: 2-3-0
                        indices.extend_from_slice(&[
                            base_index, base_index + 1, base_index + 2,
                            base_index + 2, base_index + 3, base_index,
                        ]);
                    }
                }
            }
        }

        (opaque_vertices, opaque_indices, transparent_vertices, transparent_indices)
    }
}

/// Generate tree at world position using world for cross-chunk placement
/// This allows trees to span chunk boundaries properly
fn generate_tree(
    world: &mut World,
    world_x: i32,
    surface_y: i32,
    world_z: i32,
    tree_height: i32,
    log_block: BlockType,
    leaves_block: BlockType
) {
    let air_block = BlockType::air();

    // Place trunk vertically
    for trunk_y in 0..tree_height {
        let y = surface_y + trunk_y;
        if y < CHUNK_HEIGHT as i32 {
            world.set_block(world_x, y, world_z, log_block);
        }
    }

    // Place leaves in a 5x5x4 blob around the top of the trunk
    let leaf_start_y = surface_y + tree_height - 2;
    for leaf_y in 0..4 {
        for leaf_x in -2..=2 {
            for leaf_z in -2..=2 {
                let block_x = world_x + leaf_x;
                let block_z = world_z + leaf_z;
                let block_y = leaf_start_y + leaf_y;

                if block_y >= 0 && block_y < CHUNK_HEIGHT as i32 {
                    // Skip center trunk blocks
                    if leaf_x == 0 && leaf_z == 0 && leaf_y < 2 {
                        continue;
                    }

                    // Create spherical-ish leaf shape using distance formula
                    let dist = (leaf_x * leaf_x + leaf_z * leaf_z + (leaf_y - 2) * (leaf_y - 2)) as f32;
                    if dist < 8.0 {
                        let current = world.get_block(block_x, block_y, block_z);
                        if current == air_block {
                            world.set_block(block_x, block_y, block_z, leaves_block);
                        }
                    }
                }
            }
        }
    }
}
