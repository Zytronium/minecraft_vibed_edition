use winit::{
    event::*,
    event_loop::{EventLoop, ActiveEventLoop},
    window::Window,
    keyboard::{KeyCode, PhysicalKey},
    application::ApplicationHandler,
};
use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use noise::{NoiseFn, Perlin};

// Chunk dimensions
const CHUNK_WIDTH: usize = 16;   // Horizontal size (X and Z)
const CHUNK_HEIGHT: usize = 128; // Vertical size (Y) - like Minecraft 1.18+

// Number of chunks in each horizontal direction (16x16 = 256 total chunks)
const WORLD_SIZE_CHUNKS: i32 = 16;

/// Vertex data sent to the GPU for rendering
/// Each vertex represents a corner of a triangle in the mesh
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],      // 3D position in world space
    tex_coords: [f32; 2],    // UV coordinates for texture mapping (0.0 to 1.0)
    brightness: f32,         // Lighting value (darker for bottom/sides, brighter for top)
    texture_index: u32,      // Which texture to use from our texture array
}

impl Vertex {
    /// Describes the memory layout of vertex data for the GPU
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position attribute (location 0)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinate attribute (location 1)
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Brightness attribute (location 2)
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                // Texture index attribute (location 3)
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// First-person camera for navigating the world
struct Camera {
    position: Vec3,     // Camera position in world space
    yaw: f32,           // Horizontal rotation (left/right)
    pitch: f32,         // Vertical rotation (up/down)
    aspect: f32,        // Aspect ratio (width/height)
    fov: f32,           // Field of view in radians
    near: f32,          // Near clipping plane
    far: f32,           // Far clipping plane
}

impl Camera {
    fn new(aspect: f32) -> Self {
        Self {
            // Start camera in the middle of the world, elevated above terrain
            position: Vec3::new(
                (WORLD_SIZE_CHUNKS * CHUNK_WIDTH as i32 / 2) as f32,
                80.0,  // Above the surface (surface is around Y 64)
                (WORLD_SIZE_CHUNKS * CHUNK_WIDTH as i32 / 2) as f32
            ),
            yaw: -90.0_f32.to_radians(),  // Face forward (negative Z)
            pitch: 0.0,
            aspect,
            fov: 70.0_f32.to_radians(),
            near: 0.1,
            far: 500.0,  // Increased to see far chunks
        }
    }

    /// Calculate the combined view-projection matrix for the shader
    fn get_view_projection(&self) -> [[f32; 4]; 4] {
        // Calculate camera direction from yaw and pitch
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let front = Vec3::new(
            cos_pitch * cos_yaw,
            sin_pitch,
            cos_pitch * sin_yaw,
        ).normalize();

        // Create view matrix (camera transform)
        let view = Mat4::look_at_rh(
            self.position,
            self.position + front,
            Vec3::Y,
        );

        // Create projection matrix (perspective)
        let proj = Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far);

        // Combine and return as 2D array for shader
        (proj * view).to_cols_array_2d()
    }

    /// Get forward direction vector (ignoring pitch, for WASD movement)
    fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, sin_yaw).normalize()
    }

    /// Get right direction vector (for A/D strafing)
    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }
}

/// Different types of blocks in the world
#[derive(Clone, Copy, PartialEq)]
enum BlockType {
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
    fn get_texture_indices(&self, face: usize) -> u32 {
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
}

/// A 16x128x16 section of the world
struct Chunk {
    blocks: [[[BlockType; CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
    offset_x: i32,  // Chunk position in world (X * CHUNK_WIDTH)
    offset_z: i32,  // Chunk position in world (Z * CHUNK_WIDTH)
}

/// Smooth interpolation function (smoothstep) for blending values
/// Returns 0 when x <= edge0, 1 when x >= edge1, smooth curve in between
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Chunk {
    /// Generate a new chunk with procedural terrain at the given chunk coordinates
    fn new(chunk_x: i32, chunk_z: i32) -> Self {
        let mut blocks = [[[BlockType::Air; CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];

        // Initialize multiple Perlin noise generators with different seeds for variety
        let terrain_perlin = Perlin::new(42);      // Primary terrain shape
        let biome_perlin = Perlin::new(1337);      // Biome selection
        let cave_perlin = Perlin::new(9999);       // 3D cave system
        let detail_perlin = Perlin::new(7777);     // Fine surface details
        let tree_perlin = Perlin::new(5555);       // Tree placement

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

                // Temperature and moisture create different biome types
                // let temperature = biome_perlin.get([nx * 0.5, nz * 0.5]);
                // let moisture = biome_perlin.get([nx * 0.5 + 100.0, nz * 0.5 + 100.0]);

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
                // Map biome_noise from [-1, 1] to blend factors

                // Calculate blend weights using smooth transitions
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
                    blocks[x][y][z] = if y < 3 {
                        // Bedrock layer at the very bottom (Y 0-2)
                        BlockType::Stone

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
                            let surface_cave_noise = cave_perlin.get([
                                world_x / 40.0,
                                world_z / 40.0
                            ]);
                            surface_cave_noise > 0.7
                        } else {
                            false
                        };

                        if is_cave || surface_cave {
                            BlockType::Air
                        } else {
                            // === SURFACE LAYER COMPOSITION ===
                            if depth == 1 {
                                // Top layer is grass
                                BlockType::Grass
                            } else if depth <= 5 {
                                // Next 2-5 layers are dirt
                                BlockType::Dirt
                            } else {
                                // Everything deeper is stone
                                BlockType::Stone
                            }
                        }
                    } else {
                        // Above surface - air
                        BlockType::Air
                    };
                }

                // === TREE GENERATION ===
                // Only place trees on grass blocks at appropriate elevations
                let surface_block = blocks[x][surface_height as usize - 1][z];
                if surface_block == BlockType::Grass && surface_height > 50 && surface_height < 100 {
                    // Use noise to randomly place trees (higher values = tree spawn)
                    let tree_noise = tree_perlin.get([world_x / 4.0, world_z / 4.0]);

                    // Trees are less common in mountains, more common in plains/hills
                    let tree_density = if mountain_weight > 0.5 {
                        0.85  // Sparse trees in mountains
                    } else if plains_weight > 0.5 {
                        0.75  // More trees in plains
                    } else {
                        0.78  // Medium density in other biomes
                    };

                    if tree_noise > tree_density {
                        // Generate a simple tree (4-6 blocks tall trunk, leaves on top)
                        let tree_height = 5 + ((tree_noise * 100.0) as i32 % 2);

                        // Place trunk
                        for trunk_y in 0..tree_height {
                            let y = surface_height as usize + trunk_y as usize;
                            if y < CHUNK_HEIGHT {
                                blocks[x][y][z] = BlockType::Log;
                            }
                        }

                        // Place leaves in a 5x5x4 blob around the top of the trunk
                        let leaf_start_y = surface_height + tree_height - 2;
                        for leaf_y in 0..4 {
                            for leaf_x in -2..=2 {
                                for leaf_z in -2..=2 {
                                    let block_x = x as i32 + leaf_x;
                                    let block_z = z as i32 + leaf_z;
                                    let block_y = leaf_start_y + leaf_y;

                                    // Check bounds
                                    if block_x >= 0 && block_x < CHUNK_WIDTH as i32 &&
                                       block_z >= 0 && block_z < CHUNK_WIDTH as i32 &&
                                       block_y >= 0 && block_y < CHUNK_HEIGHT as i32 {

                                        // Skip center trunk blocks
                                        if leaf_x == 0 && leaf_z == 0 && leaf_y < 2 {
                                            continue;
                                        }

                                        // Create spherical-ish leaf shape
                                        let dist = (leaf_x * leaf_x + leaf_z * leaf_z +
                                                   (leaf_y - 2) * (leaf_y - 2)) as f32;
                                        if dist < 8.0 {
                                            let current = blocks[block_x as usize][block_y as usize][block_z as usize];
                                            if current == BlockType::Air {
                                                blocks[block_x as usize][block_y as usize][block_z as usize] = BlockType::Leaves;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            blocks,
            offset_x: world_offset_x,
            offset_z: world_offset_z,
        }
    }

    /// Get block at local chunk coordinates, returning Air if out of bounds
    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x < 0 || y < 0 || z < 0 ||
            x >= CHUNK_WIDTH as i32 || y >= CHUNK_HEIGHT as i32 || z >= CHUNK_WIDTH as i32 {
            return BlockType::Air;
        }
        self.blocks[x as usize][y as usize][z as usize]
    }

    /// Generate mesh geometry for this chunk
    /// Only creates faces that are visible (adjacent to air blocks)
    fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Iterate through every block in the chunk
        for x in 0..CHUNK_WIDTH as i32 {
            for y in 0..CHUNK_HEIGHT as i32 {
                for z in 0..CHUNK_WIDTH as i32 {
                    let block = self.get_block(x, y, z);

                    // Skip air blocks - nothing to render
                    if block == BlockType::Air {
                        continue;
                    }

                    // World position of this block
                    let pos = [
                        (self.offset_x + x) as f32,
                        y as f32,
                        (self.offset_z + z) as f32
                    ];

                    // Check all 6 faces of the block
                    // Format: (offset to check, face_id)
                    let faces = [
                        ([-1, 0, 0], 0), // Left face (-X)
                        ([1, 0, 0], 1),  // Right face (+X)
                        ([0, -1, 0], 2), // Bottom face (-Y)
                        ([0, 1, 0], 3),  // Top face (+Y)
                        ([0, 0, -1], 4), // Back face (-Z)
                        ([0, 0, 1], 5),  // Front face (+Z)
                    ];

                    for (offset, face_id) in faces {
                        let neighbor = self.get_block(x + offset[0], y + offset[1], z + offset[2]);

                        // Render face if neighbor is air, or if current block is leaves and neighbor is not leaves
                        // This makes leaves see-through to other blocks but not to each other
                        let should_render = neighbor == BlockType::Air ||
                                          (block == BlockType::Leaves && neighbor != BlockType::Leaves);

                        if !should_render {
                            continue;
                        }

                        // Generate the 4 vertices for this face
                        let base_index = vertices.len() as u32;
                        let texture_index = block.get_texture_indices(face_id);
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

        (vertices, indices)
    }
}

/// Generate the 4 vertices for a single cube face
/// Returns vertices in counter-clockwise order for proper backface culling
fn get_face_vertices(pos: [f32; 3], face: usize, texture_index: u32) -> [Vertex; 4] {
    let [x, y, z] = pos;

    // Different brightness for different faces (Minecraft-style shading)
    let brightness = match face {
        3 => 1.0,       // Top - brightest (full sunlight)
        4 | 5 => 0.8,   // Front/Back - medium-bright
        0 | 1 => 0.6,   // Left/Right - medium-dark
        2 => 0.5,       // Bottom - darkest
        _ => 1.0,
    };

    // UV coordinates map texture onto face
    // (0,0) = top-left, (1,1) = bottom-right in texture space
    match face {
        0 => [ // Left face (-X)
            Vertex { position: [x, y, z], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z], tex_coords: [0.0, 0.0], brightness, texture_index },
        ],
        1 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
        ],
        2 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y, z], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
        ],
        3 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [1.0, 1.0], brightness, texture_index },
        ],
        4 => [ // Back face (-Z)
            Vertex { position: [x, y, z], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [0.0, 1.0], brightness, texture_index },
        ],
        5 => [ // Front face (+Z)
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
        ],
        _ => unreachable!(),
    }
}

/// Load a texture from disk and upload to GPU
fn load_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &str,
) -> wgpu::Texture {
    // Load image from file and convert to RGBA
    let img = image::open(path).unwrap().to_rgba8();
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    // Create GPU texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(path),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload image data to GPU
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        size,
    );

    texture
}

/// Main rendering state - holds all GPU resources and game state
struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    keys_pressed: std::collections::HashSet<KeyCode>,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // Create WebGPU instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface for rendering to the window
        let surface = instance.create_surface(window.clone()).unwrap();

        // Request GPU adapter
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        // Request device and queue with required features
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                // Enable features needed for texture arrays with dynamic indexing
                required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ).await.unwrap();

        // Configure surface for rendering
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Load all block textures - ORDER MATTERS! Must match get_texture_indices()
        let texture_paths = [
            "assets/textures/grass_top.png",        // Index 0
            "assets/textures/grass_bottom.png",     // Index 1
            "assets/textures/grass_side.png",       // Index 2
            "assets/textures/dirt.png",             // Index 3
            "assets/textures/stone.png",            // Index 4
            "assets/textures/log_top-bottom.png",   // Index 5
            "assets/textures/log_side.png",         // Index 6
            "assets/textures/leaves.png",           // Index 7
        ];

        let textures: Vec<_> = texture_paths.iter()
            .map(|path| load_texture(&device, &queue, path))
            .collect();

        let texture_views: Vec<_> = textures.iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();

        // Create sampler for texture filtering (Nearest = pixelated Minecraft look)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout for textures
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: Some(std::num::NonZeroU32::new(8).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

        // Bind textures to the layout
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&texture_views.iter().collect::<Vec<_>>()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

        // Initialize camera
        let camera = Camera::new(size.width as f32 / size.height as f32);

        // Create buffer to hold camera matrix on GPU
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera.get_view_projection()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout for camera
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        // Bind camera buffer to the layout
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Load shader code
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create render pipeline layout
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline (defines how to draw)
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),  // Enable alpha blending for transparency
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),  // Don't render back faces
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,  // Closer objects occlude farther ones
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Generate all chunks and combine into single mesh
        println!("Generating {}x{} chunks...", WORLD_SIZE_CHUNKS, WORLD_SIZE_CHUNKS);
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();

        for chunk_x in 0..WORLD_SIZE_CHUNKS {
            for chunk_z in 0..WORLD_SIZE_CHUNKS {
                let chunk = Chunk::new(chunk_x, chunk_z);
                let (vertices, indices) = chunk.generate_mesh();

                // Offset indices for combined mesh
                let index_offset = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                all_indices.extend(indices.iter().map(|&i| i + index_offset));
            }
        }

        println!("Generated {} vertices, {} triangles",
                 all_vertices.len(),
                 all_indices.len() / 3
        );

        // Upload mesh to GPU
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = all_indices.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera,
            camera_buffer,
            camera_bind_group,
            texture_bind_group,
            keys_pressed: std::collections::HashSet::new(),
            mouse_pressed: false,
            last_mouse_pos: None,
            window,
        }
    }

    /// Handle window resize
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

    /// Handle input events (keyboard, mouse)
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state,
                    ..
                },
                ..
            } => {
                // Track which keys are currently pressed
                let is_pressed = *state == ElementState::Pressed;
                if is_pressed {
                    self.keys_pressed.insert(*key);
                } else {
                    self.keys_pressed.remove(key);
                }
                true
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                // Right mouse button enables mouse look
                self.mouse_pressed = *state == ElementState::Pressed;
                if self.mouse_pressed {
                    let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                    self.window.set_cursor_visible(false);
                } else {
                    let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    self.window.set_cursor_visible(true);
                    self.last_mouse_pos = None;
                }
                true
            }
            _ => false,
        }
    }

    /// Update game state (camera movement)
    fn update(&mut self, dt: f32) {
        let speed = 10.0 * dt;  // Movement speed (blocks per second)

        // WASD movement
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.camera.position += self.camera.forward() * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.camera.position -= self.camera.forward() * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.camera.position -= self.camera.right() * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.camera.position += self.camera.right() * speed;
        }

        // Space/Shift for up/down
        if self.keys_pressed.contains(&KeyCode::Space) {
            self.camera.position.y += speed;
        }
        if self.keys_pressed.contains(&KeyCode::ShiftLeft) {
            self.camera.position.y -= speed;
        }

        // Update camera matrix on GPU
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.get_view_projection()]),
        );
    }

    /// Handle mouse movement for camera look
    fn mouse_moved(&mut self, position: (f64, f64)) {
        if !self.mouse_pressed {
            return;
        }

        if let Some(last_pos) = self.last_mouse_pos {
            let dx = (position.0 - last_pos.0) as f32;
            let dy = (position.1 - last_pos.1) as f32;

            // Mouse sensitivity
            let sensitivity = 0.002;
            self.camera.yaw += dx * sensitivity;
            self.camera.pitch -= dy * sensitivity;

            // Clamp pitch to prevent flipping upside down
            self.camera.pitch = self.camera.pitch.clamp(
                -89.0_f32.to_radians(),
                89.0_f32.to_radians()
            );
        }

        self.last_mouse_pos = Some(position);
    }

    /// Render a frame
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Get next frame from swap chain
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth buffer for this frame
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder to record rendering commands
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            // Begin render pass
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,  // Sky blue
                            g: 0.7,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set pipeline and bind groups
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);

            // Set vertex and index buffers
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw all triangles
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // Submit commands and present frame
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Application state for winit event loop
struct App {
    state: Option<State>,
    last_render_time: std::time::Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Minecraft: Vibed Edition")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.state = Some(pollster::block_on(State::new(window)));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        if let Some(state) = &mut self.state {
            if !state.input(&event) {
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        state.mouse_moved((position.x, position.y));
                    }
                    WindowEvent::RedrawRequested => {
                        let now = std::time::Instant::now();
                        let dt = (now - self.last_render_time).as_secs_f32();
                        self.last_render_time = now;

                        state.update(dt);
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                            Err(e) => eprintln!("{:?}", e),
                        }
                        state.window.request_redraw();
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let mut app = App {
        state: None,
        last_render_time: std::time::Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}
