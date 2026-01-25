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
use noise::{NoiseFn, Perlin, Seedable};

// Size of each chunk in blocks (16x16x16)
const CHUNK_SIZE: usize = 16;

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
            // Start camera in the middle of the world, elevated
            position: Vec3::new(
                (WORLD_SIZE_CHUNKS * CHUNK_SIZE as i32 / 2) as f32,
                20.0,
                (WORLD_SIZE_CHUNKS * CHUNK_SIZE as i32 / 2) as f32
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
    Air,    // Empty space (not rendered)
    Grass,  // Grass block with different textures per face
    Dirt,   // Dirt block
    Stone,  // Stone block
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
            BlockType::Dirt => 3,  // All faces use dirt.png
            BlockType::Stone => 4, // All faces use stone.png
        }
    }
}

/// A 16x16x16 section of the world
struct Chunk {
    blocks: [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    offset_x: i32,  // Chunk position in world (X * CHUNK_SIZE)
    offset_z: i32,  // Chunk position in world (Z * CHUNK_SIZE)
}

impl Chunk {
    /// Generate a new chunk with procedural terrain at the given chunk coordinates
    fn new(chunk_x: i32, chunk_z: i32) -> Self {
        let mut blocks = [[[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];

        // Initialize Perlin noise generator with a seed
        let perlin = Perlin::new(42);

        // World offset for this chunk
        let world_offset_x = chunk_x * CHUNK_SIZE as i32;
        let world_offset_z = chunk_z * CHUNK_SIZE as i32;

        // Generate terrain for each column in the chunk
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                // Calculate world coordinates for noise sampling
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Normalize coordinates for noise (smaller values = larger features)
                let nx = world_x / 32.0;
                let nz = world_z / 32.0;

                // Layer 1: Large-scale terrain features (mountains and valleys)
                let base = perlin.get([nx * 1.5, nz * 1.5]) * 10.0;

                // Layer 2: Medium-scale hills
                let hills = perlin.get([nx * 4.0, nz * 4.0]) * 4.0;

                // Layer 3: Small details
                let detail = perlin.get([nx * 8.0, nz * 8.0]) * 1.5;

                // Combine layers and clamp to valid range
                let height = (7.0 + base + hills + detail).max(3.0).min(14.0) as i32;

                // Fill column from bottom to surface height
                for y in 0..CHUNK_SIZE {
                    blocks[x][y][z] = if y < height as usize {
                        let depth = height as usize - y;

                        // Top layer is grass
                        if depth == 1 {
                            BlockType::Grass
                        // Next 3 layers are dirt
                        } else if depth <= 4 {
                            BlockType::Dirt
                        // Everything below is stone (with potential caves)
                        } else {
                            // 3D noise for cave generation
                            let cave_noise = perlin.get([
                                world_x / 12.0,
                                y as f64 / 8.0,
                                world_z / 12.0
                            ]);

                            // Create caves where noise is high, but not near surface
                            if cave_noise > 0.55 && y > 2 && depth > 5 {
                                BlockType::Air
                            } else {
                                BlockType::Stone
                            }
                        }
                    } else {
                        BlockType::Air
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

    /// Get block at local chunk coordinates, returning Air if out of bounds
    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x < 0 || y < 0 || z < 0 ||
            x >= CHUNK_SIZE as i32 || y >= CHUNK_SIZE as i32 || z >= CHUNK_SIZE as i32 {
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
        for x in 0..CHUNK_SIZE as i32 {
            for y in 0..CHUNK_SIZE as i32 {
                for z in 0..CHUNK_SIZE as i32 {
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
                        // Only create face if neighbor is air (face is visible)
                        if self.get_block(x + offset[0], y + offset[1], z + offset[2]) != BlockType::Air {
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

        // Load all block textures
        let texture_paths = [
            "assets/textures/grass_top.png",
            "assets/textures/grass_bottom.png",
            "assets/textures/grass_side.png",
            "assets/textures/dirt.png",
            "assets/textures/stone.png",
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
                    count: Some(std::num::NonZeroU32::new(5).unwrap()),
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
                    blend: Some(wgpu::BlendState::REPLACE),
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
