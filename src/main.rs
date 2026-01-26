mod block;
mod camera;
mod vertex;
mod world;
mod loading;
mod menu;
mod models;

use winit::{
    event::*,
    event_loop::{EventLoop, ActiveEventLoop},
    window::Window,
    keyboard::{KeyCode, PhysicalKey},
    application::ApplicationHandler,
};
use wgpu::util::DeviceExt;
use std::sync::Arc;

use camera::Camera;
use vertex::Vertex;
use world::{World, ChunkMeshData};
use loading::LoadingScreen;
use menu::{MainMenu, WorldSize};
use std::sync::mpsc;

/// Game mode - Creative or Spectator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameMode {
    Creative,   // Gravity, collision, block interaction
    Spectator,  // No gravity, no collision, fly through blocks
}

/// Player physics state for Creative mode
struct PlayerState {
    velocity: glam::Vec3,
    on_ground: bool,
    is_flying: bool,
    physics_enabled: bool,  // Disable physics during world generation
}

impl PlayerState {
    fn new() -> Self {
        Self {
            velocity: glam::Vec3::ZERO,
            on_ground: false,
            is_flying: false,
            physics_enabled: false,  // Start disabled until we find ground
        }
    }
}

// Embed all textures at compile time
const GRASS_TOP: &[u8] = include_bytes!("../assets/textures/grass_top.png");
const GRASS_BOTTOM: &[u8] = include_bytes!("../assets/textures/grass_bottom.png");
const GRASS_SIDE: &[u8] = include_bytes!("../assets/textures/grass_side.png");
const DIRT: &[u8] = include_bytes!("../assets/textures/dirt.png");
const STONE: &[u8] = include_bytes!("../assets/textures/stone.png");
const LOG_TOP_BOTTOM: &[u8] = include_bytes!("../assets/textures/log_top-bottom.png");
const LOG_SIDE: &[u8] = include_bytes!("../assets/textures/log_side.png");
const LEAVES: &[u8] = include_bytes!("../assets/textures/leaves.png");
const BEDROCK: &[u8] = include_bytes!("../assets/textures/bedrock.png");
const NOT_FOUND: &[u8] = include_bytes!("../assets/textures/not_found.png");

// Embed JSON data files at compile time
const BLOCKS_JSON: &str = include_str!("blocks.json");
const ITEMS_JSON: &str = include_str!("items.json");
const CRAFTING_JSON: &str = include_str!("crafting.json");

/// GPU buffers for a single chunk's mesh data
/// Storing meshes per-chunk avoids hitting GPU buffer size limits
struct ChunkMesh {
    opaque_vertex_buffer: wgpu::Buffer,
    opaque_index_buffer: wgpu::Buffer,
    opaque_num_indices: u32,
    transparent_vertex_buffer: wgpu::Buffer,
    transparent_index_buffer: wgpu::Buffer,
    transparent_num_indices: u32,
    chunk_x: i32,
    chunk_z: i32,
}

/// Main rendering state - holds all GPU resources and game state
struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline_opaque: wgpu::RenderPipeline,
    render_pipeline_transparent: wgpu::RenderPipeline,
    chunk_meshes: Vec<ChunkMesh>,  // Separate buffers for each chunk
    world: World,  // Keep world for block modifications
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    keys_pressed: std::collections::HashSet<KeyCode>,
    window: Arc<Window>,
    game_mode: GameMode,
    player_state: PlayerState,
}

impl State {
    async fn new(window: Arc<Window>, world_size: WorldSize, instance: wgpu::Instance, surface: wgpu::Surface<'static>) -> Self {
        let size = window.inner_size();

        // Load game data from embedded JSON
        let blocks_data: models::BlocksData = serde_json::from_str(BLOCKS_JSON)
            .expect("Failed to parse blocks.json");
        let items_data: models::ItemsData = serde_json::from_str(ITEMS_JSON)
            .expect("Failed to parse items.json");
        let recipes_data: models::RecipesData = serde_json::from_str(CRAFTING_JSON)
            .expect("Failed to parse crafting.json");

        println!("Loaded {} blocks, {} items, {} recipes",
                 blocks_data.blocks.len(),
                 items_data.items.len(),
                 recipes_data.recipes.len());

        // Build texture map from blocks
        let texture_map = build_texture_map(&blocks_data.blocks);
        println!("Found {} unique textures", texture_map.len());

        // Initialize block registry
        let mut block_registry = block::BlockRegistry::new();
        block_registry.load_blocks(blocks_data.blocks);
        block_registry.build_texture_indices(&texture_map);
        let block_registry = Arc::new(block_registry);

        // Request GPU adapter
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        // Request device and queue with required features for texture arrays
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
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

        // Load textures dynamically based on texture_map
        let textures = load_textures_from_map(&device, &queue, &texture_map);
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

        // Create bind group layout for textures - dynamic array size
        let texture_count = texture_map.len().max(1) as u32;
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
                    count: Some(std::num::NonZeroU32::new(texture_count).unwrap()),
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
        let mut camera = Camera::new(size.width as f32 / size.height as f32, world_size.get_chunks());

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

        // Opaque pipeline - with depth writes enabled
        let render_pipeline_opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Opaque Pipeline"),
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
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Transparent pipeline - without depth writes
        let render_pipeline_transparent = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Transparent Pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,  // Don't write depth for transparent
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Generate world with loading screen - NON-BLOCKING
        println!("Generating world...");
        let generation_result = {
            // Create loading screen
            let mut loading_screen = LoadingScreen::new(&device, &queue, config.format);

            // Channels for progress updates and completion signal
            let (progress_tx, progress_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();

            let world_size_chunks = world_size.get_chunks();
            let registry_clone = Arc::clone(&block_registry);

            // Spawn world generation in a separate thread
            std::thread::spawn(move || {
                let result = World::generate(world_size_chunks, registry_clone, move |current, total| {
                    let _ = progress_tx.send((current, total));
                });
                let _ = done_tx.send(result);
            });

            // Keep window responsive during generation
            let start_time = std::time::Instant::now();
            let mut last_update = start_time;
            let mut num_indices = 6; // Start with just background

            // Render loading screen with progress updates
            loop {
                // Update progress from background thread
                while let Ok((current, total)) = progress_rx.try_recv() {
                    let progress = current as f32 / total as f32;
                    num_indices = loading_screen.update_progress(&queue, progress, current, total);
                }

                // Limit render rate to 60fps to reduce CPU usage
                let now = std::time::Instant::now();
                if now.duration_since(last_update).as_millis() >= 16 {
                    last_update = now;

                    // Render the loading screen
                    match surface.get_current_texture() {
                        Ok(output) => {
                            let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                            loading_screen.render(&view, &device, &queue, num_indices);
                            output.present();
                        }
                        Err(wgpu::SurfaceError::Lost) => {
                            surface.configure(&device, &config);
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            // Just skip this frame
                        }
                        Err(e) => {
                            eprintln!("Surface error during loading: {:?}", e);
                        }
                    }
                }

                // Check if generation is complete
                if let Ok((world_data, chunk_mesh_data)) = done_rx.try_recv() {
                    // Render final loading screen at 100%
                    let total_chunks = (world_size_chunks * world_size_chunks) as usize;
                    num_indices = loading_screen.update_progress(&queue, 1.0, total_chunks, total_chunks);
                    if let Ok(output) = surface.get_current_texture() {
                        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        loading_screen.render(&view, &device, &queue, num_indices);
                        output.present();
                    }
                    // Small delay so user can see 100%
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    break (world_data, chunk_mesh_data);
                }

                // Small sleep to prevent busy-waiting
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };

        // Destructure the tuple
        let (world_data, chunk_meshes) = generation_result;

        // Convert per-chunk mesh data into GPU buffers
        // IMPORTANT: Each chunk gets its own set of buffers to avoid hitting GPU limits
        let mut gpu_chunk_meshes = Vec::new();
        let mut total_opaque_verts = 0;
        let mut total_transparent_verts = 0;

        for chunk_data in chunk_meshes {
            total_opaque_verts += chunk_data.opaque_vertices.len();
            total_transparent_verts += chunk_data.transparent_vertices.len();

            // Create GPU buffers for this chunk's opaque geometry
            let opaque_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Chunk ({},{}) Opaque Vertex Buffer", chunk_data.chunk_x, chunk_data.chunk_z)),
                contents: bytemuck::cast_slice(&chunk_data.opaque_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let opaque_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Chunk ({},{}) Opaque Index Buffer", chunk_data.chunk_x, chunk_data.chunk_z)),
                contents: bytemuck::cast_slice(&chunk_data.opaque_indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            // Create GPU buffers for this chunk's transparent geometry
            let transparent_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Chunk ({},{}) Transparent Vertex Buffer", chunk_data.chunk_x, chunk_data.chunk_z)),
                contents: bytemuck::cast_slice(&chunk_data.transparent_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let transparent_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Chunk ({},{}) Transparent Index Buffer", chunk_data.chunk_x, chunk_data.chunk_z)),
                contents: bytemuck::cast_slice(&chunk_data.transparent_indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            gpu_chunk_meshes.push(ChunkMesh {
                opaque_vertex_buffer,
                opaque_index_buffer,
                opaque_num_indices: chunk_data.opaque_indices.len() as u32,
                transparent_vertex_buffer,
                transparent_index_buffer,
                transparent_num_indices: chunk_data.transparent_indices.len() as u32,
                chunk_x: chunk_data.chunk_x,
                chunk_z: chunk_data.chunk_z,
            });
        }

        println!("Generated {} chunks with {} opaque vertices, {} transparent vertices",
                 gpu_chunk_meshes.len(), total_opaque_verts, total_transparent_verts);

        // Lock cursor for FPS-style mouse look (allows infinite movement)
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
        window.set_cursor_visible(false);

        // Find actual ground level at spawn position and place player on top
        let spawn_x = (world_size.get_chunks() * world::CHUNK_WIDTH as i32 / 2) as i32;
        let spawn_z = (world_size.get_chunks() * world::CHUNK_WIDTH as i32 / 2) as i32;
        let mut spawn_y = None;

        println!("Searching for ground at spawn location ({}, ?, {})", spawn_x, spawn_z);

        // Scan downward from top of world to find first non-air block
        for y in (0..world::CHUNK_HEIGHT as i32).rev() {
            let block = world_data.get_block(spawn_x, y, spawn_z);
            if block != block::BlockType::air() {
                spawn_y = Some(y + 1); // Spawn with feet 1 block above ground (standing on top of it)
                println!("Found ground block at Y={}, spawning player feet at Y={}", y, y + 1);
                break;
            }
        }

        // Fallback if no ground found (shouldn't happen, but just in case)
        let spawn_y = spawn_y.unwrap_or_else(|| {
            eprintln!("ERROR: No ground found at spawn location ({}, {})! Using fallback Y=80", spawn_x, spawn_z);
            eprintln!("This suggests the world data may not be properly initialized.");
            80
        });

        // Update camera position to actual spawn point (add eye height)
        camera.position = glam::Vec3::new(spawn_x as f32 + 0.5, spawn_y as f32 + 1.62, spawn_z as f32 + 0.5);
        println!("===== SPAWN INFO =====");
        println!("Player spawned at: ({:.1}, {:.1}, {:.1})", camera.position.x, camera.position.y, camera.position.z);
        println!("Feet position: Y={}", spawn_y);
        println!("Ground level: Y={}", spawn_y - 1);
        println!("======================");

        let mut player_state = PlayerState::new();
        player_state.physics_enabled = true; // Enable physics now that world is ready
        player_state.on_ground = true; // Start on ground to prevent immediate falling

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline_opaque,
            render_pipeline_transparent,
            chunk_meshes: gpu_chunk_meshes,
            world: world_data,
            camera,
            camera_buffer,
            camera_bind_group,
            texture_bind_group,
            keys_pressed: std::collections::HashSet::new(),
            window,
            game_mode: GameMode::Creative,
            player_state,
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

                    // Toggle game mode with G key
                    if *key == KeyCode::KeyG {
                        self.game_mode = match self.game_mode {
                            GameMode::Creative => GameMode::Spectator,
                            GameMode::Spectator => GameMode::Creative,
                        };
                        println!("Game mode: {:?}", self.game_mode);
                    }

                    // Toggle flying in Creative mode with F key
                    if *key == KeyCode::KeyF && self.game_mode == GameMode::Creative {
                        self.player_state.is_flying = !self.player_state.is_flying;
                        if self.player_state.is_flying {
                            self.player_state.velocity.y = 0.0; // Stop falling when starting to fly
                        }
                        println!("Flying: {}", self.player_state.is_flying);
                    }

                    // Release cursor with Escape key
                    if *key == KeyCode::Escape {
                        let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                        self.window.set_cursor_visible(true);
                        println!("Cursor released - press any movement key to recapture");
                    }

                    // Recapture cursor when any movement key is pressed
                    if matches!(key, KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD | KeyCode::Space | KeyCode::ShiftLeft) {
                        let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                            .or_else(|_| self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
                        self.window.set_cursor_visible(false);
                    }
                } else {
                    self.keys_pressed.remove(key);
                }
                true
            }
            WindowEvent::MouseInput {
                state,
                button,
                ..
            } => {
                if *state == ElementState::Pressed {
                    match button {
                        MouseButton::Left => {
                            // Break block
                            self.break_block();
                        }
                        MouseButton::Right => {
                            // Place block
                            self.place_block();
                        }
                        _ => {}
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Update game state (camera movement, physics)
    fn update(&mut self, dt: f32) {
        match self.game_mode {
            GameMode::Spectator => {
                // Spectator mode - free flight, no physics
                let speed = 10.0 * dt;

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
                if self.keys_pressed.contains(&KeyCode::Space) {
                    self.camera.position.y += speed;
                }
                if self.keys_pressed.contains(&KeyCode::ShiftLeft) {
                    self.camera.position.y -= speed;
                }
            }
            GameMode::Creative => {
                // Creative mode - physics-based movement
                if self.player_state.physics_enabled {
                    self.update_physics(dt);
                } else {
                    // Physics not yet enabled (world still generating) - use spectator controls
                    let speed = 10.0 * dt;

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
                    if self.keys_pressed.contains(&KeyCode::Space) {
                        self.camera.position.y += speed;
                    }
                    if self.keys_pressed.contains(&KeyCode::ShiftLeft) {
                        self.camera.position.y -= speed;
                    }
                }
            }
        }

        // Update camera matrix on GPU
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.get_view_projection()]),
        );
    }

    /// Update physics for Creative mode (gravity, collision, movement)
    fn update_physics(&mut self, dt: f32) {
        use glam::Vec3;

        // Player dimensions (Minecraft-like)
        const PLAYER_WIDTH: f32 = 0.6;
        const PLAYER_HEIGHT: f32 = 1.8;
        const PLAYER_EYE_HEIGHT: f32 = 1.62; // Eye level is slightly below top of head

        // Physics constants
        const GRAVITY: f32 = -32.0; // blocks/s² (increased from -20 for faster falling)
        const TERMINAL_VELOCITY: f32 = -78.0; // blocks/s (increased from -40)
        const JUMP_VELOCITY: f32 = 9.5; // blocks/s
        const WALK_SPEED: f32 = 4.3; // blocks/s
        const FLY_SPEED: f32 = 10.9; // blocks/s (creative flying)
        const AIR_CONTROL: f32 = 0.8; // Increased from 0.2 - good air control

        // Calculate player's feet position (camera is at eye level)
        let feet_pos = self.camera.position - Vec3::new(0.0, PLAYER_EYE_HEIGHT, 0.0);

        if self.player_state.is_flying {
            // Flying mode - simple movement in all directions
            let speed = FLY_SPEED * dt;

            let mut fly_dir = Vec3::ZERO;
            if self.keys_pressed.contains(&KeyCode::KeyW) {
                fly_dir += self.camera.forward();
            }
            if self.keys_pressed.contains(&KeyCode::KeyS) {
                fly_dir -= self.camera.forward();
            }
            if self.keys_pressed.contains(&KeyCode::KeyA) {
                fly_dir -= self.camera.right();
            }
            if self.keys_pressed.contains(&KeyCode::KeyD) {
                fly_dir += self.camera.right();
            }
            if self.keys_pressed.contains(&KeyCode::Space) {
                fly_dir.y += 1.0;
            }
            if self.keys_pressed.contains(&KeyCode::ShiftLeft) {
                fly_dir.y -= 1.0;
            }

            if fly_dir.length_squared() > 0.0 {
                fly_dir = fly_dir.normalize();
            }

            self.camera.position += fly_dir * speed;
            self.player_state.velocity = Vec3::ZERO;
            self.player_state.on_ground = false;
        } else {
            // Walking mode - apply gravity and ground physics

            // Horizontal movement
            let mut move_dir = Vec3::ZERO;
            if self.keys_pressed.contains(&KeyCode::KeyW) {
                move_dir += self.camera.forward();
            }
            if self.keys_pressed.contains(&KeyCode::KeyS) {
                move_dir -= self.camera.forward();
            }
            if self.keys_pressed.contains(&KeyCode::KeyA) {
                move_dir -= self.camera.right();
            }
            if self.keys_pressed.contains(&KeyCode::KeyD) {
                move_dir += self.camera.right();
            }

            // Normalize diagonal movement
            if move_dir.length_squared() > 0.0 {
                move_dir = move_dir.normalize();
            }

            // Check if on ground BEFORE applying movement
            self.player_state.on_ground = self.is_on_ground(feet_pos, PLAYER_WIDTH, PLAYER_HEIGHT);

            // Apply gravity
            if !self.player_state.on_ground {
                self.player_state.velocity.y += GRAVITY * dt;
                if self.player_state.velocity.y < TERMINAL_VELOCITY {
                    self.player_state.velocity.y = TERMINAL_VELOCITY;
                }
            } else {
                // On ground - no falling
                self.player_state.velocity.y = 0.0;
            }

            // Jump
            if self.keys_pressed.contains(&KeyCode::Space) && self.player_state.on_ground {
                self.player_state.velocity.y = JUMP_VELOCITY;
                self.player_state.on_ground = false;
            }

            // Calculate movement
            let horizontal_speed = if self.player_state.on_ground { WALK_SPEED } else { WALK_SPEED * AIR_CONTROL };

            // Move horizontally (with collision)
            let horizontal_velocity = move_dir * horizontal_speed * dt;
            let mut new_feet_pos = feet_pos;

            // Move X
            new_feet_pos.x += horizontal_velocity.x;
            if self.check_collision_at(new_feet_pos, PLAYER_WIDTH, PLAYER_HEIGHT) {
                new_feet_pos.x = feet_pos.x; // Cancel X movement
            }

            // Move Z
            new_feet_pos.z += horizontal_velocity.z;
            if self.check_collision_at(new_feet_pos, PLAYER_WIDTH, PLAYER_HEIGHT) {
                new_feet_pos.z = feet_pos.z; // Cancel Z movement
            }

            // Move Y (gravity/jumping)
            new_feet_pos.y += self.player_state.velocity.y * dt;
            if self.check_collision_at(new_feet_pos, PLAYER_WIDTH, PLAYER_HEIGHT) {
                new_feet_pos.y = feet_pos.y; // Cancel Y movement
                self.player_state.velocity.y = 0.0; // Stop vertical velocity
            }

            // Update camera position (eye level)
            self.camera.position = new_feet_pos + Vec3::new(0.0, PLAYER_EYE_HEIGHT, 0.0);
        }
    }

    /// Check if player bounding box collides with any solid blocks at given position
    /// Uses proper AABB-to-block collision: checks ALL blocks the bounding box overlaps
    fn check_collision_at(&self, feet_pos: glam::Vec3, width: f32, height: f32) -> bool {
        let half_width = width / 2.0;

        // Calculate the bounding box in block coordinates
        let min_x = (feet_pos.x - half_width).floor() as i32;
        let max_x = (feet_pos.x + half_width).ceil() as i32;
        let min_y = feet_pos.y.floor() as i32;
        let max_y = (feet_pos.y + height).ceil() as i32;
        let min_z = (feet_pos.z - half_width).floor() as i32;
        let max_z = (feet_pos.z + half_width).ceil() as i32;

        // Check every block that the player's bounding box overlaps
        for x in min_x..max_x {
            for y in min_y..max_y {
                for z in min_z..max_z {
                    let block = self.world.get_block(x, y, z);
                    if block != block::BlockType::air() {
                        return true; // Collision detected
                    }
                }
            }
        }

        false // No collision
    }

    /// Check if player is standing on ground
    fn is_on_ground(&self, feet_pos: glam::Vec3, width: f32, _height: f32) -> bool {
        let half_width = width / 2.0;

        // Check blocks just below the player's feet
        let check_y = (feet_pos.y - 0.01).floor() as i32;
        let min_x = (feet_pos.x - half_width).floor() as i32;
        let max_x = (feet_pos.x + half_width).ceil() as i32;
        let min_z = (feet_pos.z - half_width).floor() as i32;
        let max_z = (feet_pos.z + half_width).ceil() as i32;

        // Check all blocks under the player's base
        for x in min_x..max_x {
            for z in min_z..max_z {
                let block = self.world.get_block(x, check_y, z);
                if block != block::BlockType::air() {
                    return true;
                }
            }
        }
        false
    }

    /// Raycast to find which block the player is looking at
    fn raycast_block(&self) -> Option<(i32, i32, i32, i32, i32, i32)> {
        use glam::Vec3;

        const MAX_DISTANCE: f32 = 5.0; // blocks
        const STEP_SIZE: f32 = 0.1;

        let start = self.camera.position;
        let direction = self.camera.look_direction();

        let mut pos = start;
        let mut last_pos = start;

        for _ in 0..(MAX_DISTANCE / STEP_SIZE) as i32 {
            last_pos = pos;
            pos += direction * STEP_SIZE;

            let block_pos = (pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32);
            let block = self.world.get_block(block_pos.0, block_pos.1, block_pos.2);

            if block != block::BlockType::air() {
                // Found a block - return hit position and face normal
                let last_block_pos = (
                    last_pos.x.floor() as i32,
                    last_pos.y.floor() as i32,
                    last_pos.z.floor() as i32,
                );
                return Some((
                    block_pos.0, block_pos.1, block_pos.2,
                    last_block_pos.0, last_block_pos.1, last_block_pos.2,
                ));
            }
        }

        None
    }

    /// Break the block the player is looking at
    fn break_block(&mut self) {
        if let Some((x, y, z, _, _, _)) = self.raycast_block() {
            // Remove the block
            self.world.set_block(x, y, z, block::BlockType::air());

            // Regenerate affected chunk meshes
            self.regenerate_chunk_at_position(x, z);
        }
    }

    /// Place a block adjacent to the one the player is looking at
    fn place_block(&mut self) {
        if let Some((_, _, _, place_x, place_y, place_z)) = self.raycast_block() {
            // Check we're not placing inside the player
            let player_feet = self.camera.position - glam::Vec3::new(0.0, 1.62, 0.0);
            let player_min_x = (player_feet.x - 0.3).floor() as i32;
            let player_max_x = (player_feet.x + 0.3).ceil() as i32;
            let player_min_y = player_feet.y.floor() as i32;
            let player_max_y = (player_feet.y + 1.8).ceil() as i32;
            let player_min_z = (player_feet.z - 0.3).floor() as i32;
            let player_max_z = (player_feet.z + 0.3).ceil() as i32;

            if place_x >= player_min_x && place_x <= player_max_x &&
                place_y >= player_min_y && place_y <= player_max_y &&
                place_z >= player_min_z && place_z <= player_max_z {
                return; // Don't place block inside player
            }

            // Place dirt block (we can make this configurable later)
            let dirt_block = self.world.registry.block_type("minecraft:dirt");
            self.world.set_block(place_x, place_y, place_z, dirt_block);

            // Regenerate affected chunk meshes
            self.regenerate_chunk_at_position(place_x, place_z);
        }
    }

    /// Regenerate the mesh for the chunk containing the given world position
    fn regenerate_chunk_at_position(&mut self, world_x: i32, world_z: i32) {
        let chunk_x = world_x.div_euclid(world::CHUNK_WIDTH as i32);
        let chunk_z = world_z.div_euclid(world::CHUNK_WIDTH as i32);

        if let Some(new_mesh_data) = self.world.regenerate_chunk_mesh(chunk_x, chunk_z) {
            // Find and update the corresponding GPU mesh
            if let Some(chunk_mesh) = self.chunk_meshes.iter_mut()
                .find(|cm| cm.chunk_x == chunk_x && cm.chunk_z == chunk_z) {

                // Recreate GPU buffers with new mesh data
                chunk_mesh.opaque_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Chunk ({},{}) Opaque Vertex Buffer", chunk_x, chunk_z)),
                    contents: bytemuck::cast_slice(&new_mesh_data.opaque_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                chunk_mesh.opaque_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Chunk ({},{}) Opaque Index Buffer", chunk_x, chunk_z)),
                    contents: bytemuck::cast_slice(&new_mesh_data.opaque_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                chunk_mesh.opaque_num_indices = new_mesh_data.opaque_indices.len() as u32;

                chunk_mesh.transparent_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Chunk ({},{}) Transparent Vertex Buffer", chunk_x, chunk_z)),
                    contents: bytemuck::cast_slice(&new_mesh_data.transparent_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                chunk_mesh.transparent_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Chunk ({},{}) Transparent Index Buffer", chunk_x, chunk_z)),
                    contents: bytemuck::cast_slice(&new_mesh_data.transparent_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                chunk_mesh.transparent_num_indices = new_mesh_data.transparent_indices.len() as u32;
            }
        }
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
                            r: 0.5, g: 0.7, b: 1.0, a: 1.0,  // Sky blue
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

            // Set up common state for all chunks
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);

            // FIRST PASS: Render all opaque geometry from all chunks
            // This must be done first with depth writes enabled
            render_pass.set_pipeline(&self.render_pipeline_opaque);
            for chunk_mesh in &self.chunk_meshes {
                if chunk_mesh.opaque_num_indices > 0 {
                    render_pass.set_vertex_buffer(0, chunk_mesh.opaque_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(chunk_mesh.opaque_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..chunk_mesh.opaque_num_indices, 0, 0..1);
                }
            }

            // SECOND PASS: Render all transparent geometry from all chunks
            // This must be done second without depth writes but with depth testing
            render_pass.set_pipeline(&self.render_pipeline_transparent);
            for chunk_mesh in &self.chunk_meshes {
                if chunk_mesh.transparent_num_indices > 0 {
                    render_pass.set_vertex_buffer(0, chunk_mesh.transparent_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(chunk_mesh.transparent_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..chunk_mesh.transparent_num_indices, 0, 0..1);
                }
            }
        }

        // Submit commands and present frame
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Build texture map from block definitions
fn build_texture_map(blocks: &[models::Block]) -> std::collections::HashMap<String, u32> {
    let mut texture_map = std::collections::HashMap::new();
    let mut next_index = 0u32;

    for block in blocks {
        // Collect all unique texture names from this block
        let textures = &block.textures;

        if let Some(ref all) = textures.all {
            texture_map.entry(all.clone()).or_insert_with(|| {
                let idx = next_index;
                next_index += 1;
                idx
            });
        }

        for texture_name in [
            &textures.top,
            &textures.bottom,
            &textures.north,
            &textures.south,
            &textures.east,
            &textures.west,
        ].iter().filter_map(|t| t.as_ref()) {
            texture_map.entry(texture_name.clone()).or_insert_with(|| {
                let idx = next_index;
                next_index += 1;
                idx
            });
        }
    }

    texture_map
}

/// Load textures dynamically based on texture_map
fn load_textures_from_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_map: &std::collections::HashMap<String, u32>,
) -> Vec<wgpu::Texture> {
    // Map of embedded texture names to their bytes
    let embedded_textures: std::collections::HashMap<&str, &[u8]> = [
        ("grass_top", GRASS_TOP),
        ("grass_bottom", GRASS_BOTTOM),
        ("grass_side", GRASS_SIDE),
        ("dirt", DIRT),
        ("stone", STONE),
        ("bedrock", BEDROCK),
        ("log_top-bottom", LOG_TOP_BOTTOM),
        ("log_side", LOG_SIDE),
        ("leaves", LEAVES),
    ].iter().cloned().collect();

    // Build a reverse map: index -> texture_name (Option<&str> is Clone because &str is Copy)
    let mut index_to_name: Vec<Option<&str>> = vec![None; texture_map.len()];
    for (texture_name, &index) in texture_map {
        index_to_name[index as usize] = Some(texture_name.as_str());
    }

    // Load each texture in index order
    index_to_name.into_iter()
        .map(|name_opt| {
            let texture_name = name_opt.expect("Missing texture for index");
            if let Some(bytes) = embedded_textures.get(texture_name) {
                load_texture_from_bytes(device, queue, bytes, texture_name)
            } else {
                eprintln!("Warning: Texture '{}' not found in embedded assets, using not_found.png", texture_name);
                // Use not_found.png as fallback for missing textures
                load_texture_from_bytes(device, queue, NOT_FOUND, "not_found")
            }
        })
        .collect()
}

/// Load a texture from embedded bytes and upload to GPU
fn load_texture_from_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], label: &str) -> wgpu::Texture {
    let img = image::load_from_memory(bytes)
        .unwrap_or_else(|e| panic!("Failed to load texture {}: {}", label, e))
        .to_rgba8();
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

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

/// Application state for winit event loop
#[derive(PartialEq)]
enum GameState {
    Menu,
    Playing,
}

struct MenuContext {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    window: Arc<Window>,
    surface_format: wgpu::TextureFormat,  // Store the format to avoid re-querying
}

struct App {
    state: Option<State>,
    menu: Option<MainMenu>,
    menu_ctx: Option<MenuContext>,
    game_state: GameState,
    last_render_time: std::time::Instant,
    last_mouse_pos: Option<(f64, f64)>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() && self.menu.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Minecraft: Vibed Edition")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

            // Create temporary GPU context for menu
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })).unwrap();

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )).unwrap();

            let size = window.inner_size();
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

            let mut menu = MainMenu::new(&device, &queue, surface_format, size.width, size.height);
            let bg_indices = menu.update_geometry(&queue);

            // Render initial menu
            if let Ok(output) = surface.get_current_texture() {
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                menu.render(&view, &device, &queue, bg_indices);
                output.present();
            }

            self.menu = Some(menu);
            self.menu_ctx = Some(MenuContext {
                instance,
                surface,
                device,
                queue,
                window,
                surface_format,  // Store format for later use
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        match self.game_state {
            GameState::Menu => {
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(physical_size) => {
                        // Handle menu window resize
                        if let (Some(menu), Some(ctx)) = (&mut self.menu, &mut self.menu_ctx) {
                            menu.resize(&ctx.queue, physical_size.width, physical_size.height);

                            // Reconfigure surface using stored format
                            let config = wgpu::SurfaceConfiguration {
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                                format: ctx.surface_format,
                                width: physical_size.width,
                                height: physical_size.height,
                                present_mode: wgpu::PresentMode::Fifo,
                                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                                view_formats: vec![],
                                desired_maximum_frame_latency: 2,
                            };
                            ctx.surface.configure(&ctx.device, &config);

                            ctx.window.request_redraw();
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if let (Some(menu), Some(ctx)) = (&mut self.menu, &self.menu_ctx) {
                            let bg_indices = menu.update_geometry(&ctx.queue);
                            if let Ok(output) = ctx.surface.get_current_texture() {
                                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                output.present();
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        self.last_mouse_pos = Some((position.x, position.y));
                        if let Some(menu) = &mut self.menu {
                            if menu.check_hover(position.x, position.y) {
                                if let Some(ctx) = &self.menu_ctx {
                                    ctx.window.request_redraw();
                                }
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    } => {
                        if let Some(menu) = &mut self.menu {
                            // Use the last known mouse position for click handling
                            if let Some((x, y)) = self.last_mouse_pos {
                                if let Some(selected_size) = menu.handle_click(x, y) {
                                    println!("Starting world generation: {:?} chunks", selected_size.get_chunks());

                                    if let Some(ctx) = self.menu_ctx.take() {
                                        self.state = Some(pollster::block_on(State::new(
                                            ctx.window,
                                            selected_size,
                                            ctx.instance,
                                            ctx.surface
                                        )));
                                        self.menu = None;
                                        self.game_state = GameState::Playing;
                                    }
                                }
                            }
                        }
                    }
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            physical_key: PhysicalKey::Code(key),
                            state: ElementState::Pressed,
                            ..
                        },
                        ..
                    } => {
                        match key {
                            KeyCode::Escape => event_loop.exit(),
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            GameState::Playing => {
                if let Some(state) = &mut self.state {
                    if !state.input(&event) {
                        match event {
                            WindowEvent::CloseRequested => event_loop.exit(),
                            WindowEvent::Resized(physical_size) => {
                                state.resize(physical_size);
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
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: winit::event::DeviceId, event: winit::event::DeviceEvent) {
        // Only process mouse motion when playing (not in menu)
        if self.game_state == GameState::Playing {
            if let Some(state) = &mut self.state {
                match event {
                    winit::event::DeviceEvent::MouseMotion { delta } => {
                        // Use raw mouse delta for camera control (works even when cursor is locked)
                        let sensitivity = 0.002;
                        state.camera.yaw += delta.0 as f32 * sensitivity;
                        state.camera.pitch -= delta.1 as f32 * sensitivity;

                        // Clamp pitch to prevent flipping upside down
                        state.camera.pitch = state.camera.pitch.clamp(
                            -89.0_f32.to_radians(),
                            89.0_f32.to_radians()
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        match self.game_state {
            GameState::Menu => {
                if let Some(ctx) = &self.menu_ctx {
                    ctx.window.request_redraw();
                }
            }
            GameState::Playing => {
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let mut app = App {
        state: None,
        menu: None,
        menu_ctx: None,
        game_state: GameState::Menu,
        last_render_time: std::time::Instant::now(),
        last_mouse_pos: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
