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

const CHUNK_SIZE: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    brightness: f32,
    texture_index: u32,
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Camera {
    fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(8.0, 10.0, 8.0),
            yaw: -90.0_f32.to_radians(),
            pitch: 0.0,
            aspect,
            fov: 70.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
        }
    }

    fn get_view_projection(&self) -> [[f32; 4]; 4] {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let front = Vec3::new(
            cos_pitch * cos_yaw,
            sin_pitch,
            cos_pitch * sin_yaw,
        ).normalize();

        let view = Mat4::look_at_rh(
            self.position,
            self.position + front,
            Vec3::Y,
        );

        let proj = Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far);
        (proj * view).to_cols_array_2d()
    }

    fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, sin_yaw).normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
}

impl BlockType {
    fn get_texture_indices(&self, face: usize) -> u32 {
        match self {
            BlockType::Air => 0,
            BlockType::Grass => {
                match face {
                    3 => 0, // Top - grass_top.png
                    2 => 1, // Bottom - grass_bottom.png
                    _ => 2, // Sides - grass_side.png
                }
            }
            BlockType::Dirt => 3, // dirt.png
            BlockType::Stone => 4, // stone.png
        }
    }
}

struct Chunk {
    blocks: [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
}

impl Chunk {
    fn new() -> Self {
        let mut blocks = [[[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];

        // Generate simple terrain
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let height = 6 + ((x + z) % 3) as i32;
                for y in 0..CHUNK_SIZE {
                    blocks[x][y][z] = if y < height as usize {
                        if y == height as usize - 1 {
                            BlockType::Grass
                        } else if y > height as usize - 4 {
                            BlockType::Dirt
                        } else {
                            BlockType::Stone
                        }
                    } else {
                        BlockType::Air
                    };
                }
            }
        }

        Self { blocks }
    }

    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x < 0 || y < 0 || z < 0 ||
            x >= CHUNK_SIZE as i32 || y >= CHUNK_SIZE as i32 || z >= CHUNK_SIZE as i32 {
            return BlockType::Air;
        }
        self.blocks[x as usize][y as usize][z as usize]
    }

    fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for x in 0..CHUNK_SIZE as i32 {
            for y in 0..CHUNK_SIZE as i32 {
                for z in 0..CHUNK_SIZE as i32 {
                    let block = self.get_block(x, y, z);
                    if block == BlockType::Air {
                        continue;
                    }

                    let pos = [x as f32, y as f32, z as f32];

                    // Check each face and only add if neighbor is air
                    let faces = [
                        ([-1, 0, 0], 0), // Left
                        ([1, 0, 0], 1),  // Right
                        ([0, -1, 0], 2), // Bottom
                        ([0, 1, 0], 3),  // Top
                        ([0, 0, -1], 4), // Back
                        ([0, 0, 1], 5),  // Front
                    ];

                    for (offset, face_id) in faces {
                        if self.get_block(x + offset[0], y + offset[1], z + offset[2]) != BlockType::Air {
                            continue;
                        }

                        let base_index = vertices.len() as u32;
                        let texture_index = block.get_texture_indices(face_id);
                        let face_verts = get_face_vertices(pos, face_id, texture_index);
                        vertices.extend_from_slice(&face_verts);

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

fn get_face_vertices(pos: [f32; 3], face: usize, texture_index: u32) -> [Vertex; 4] {
    let [x, y, z] = pos;

    // Different brightness for different faces (like Minecraft)
    let brightness = match face {
        3 => 1.0,   // Top - brightest
        4 | 5 => 0.8, // Front/Back - medium-bright
        0 | 1 => 0.6, // Left/Right - medium-dark
        2 => 0.5,   // Bottom - darkest
        _ => 1.0,
    };

    match face {
        0 => [ // Left (-X) - looking from left, CCW from bottom-left
            Vertex { position: [x, y, z], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z], tex_coords: [0.0, 0.0], brightness, texture_index },
        ],
        1 => [ // Right (+X) - looking from right, CCW from bottom-left
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
        ],
        2 => [ // Bottom (-Y) - looking from bottom, CCW
            Vertex { position: [x, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y, z], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
        ],
        3 => [ // Top (+Y) - looking from top, CCW
            Vertex { position: [x, y + 1.0, z], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [1.0, 1.0], brightness, texture_index },
        ],
        4 => [ // Back (-Z) - looking from back, CCW
            Vertex { position: [x, y, z], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [0.0, 1.0], brightness, texture_index },
        ],
        5 => [ // Front (+Z) - looking from front, CCW
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
        ],
        _ => unreachable!(),
    }
}

fn load_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &str,
) -> wgpu::Texture {
    let img = image::open(path).unwrap().to_rgba8();
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

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

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

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

        // Load textures
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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

        let camera = Camera::new(size.width as f32 / size.height as f32);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera.get_view_projection()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

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

        let chunk = Chunk::new();
        let (vertices, indices) = chunk.generate_mesh();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;

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

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

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

    fn update(&mut self, dt: f32) {
        let speed = 5.0 * dt;

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

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.get_view_projection()]),
        );
    }

    fn mouse_moved(&mut self, position: (f64, f64)) {
        if !self.mouse_pressed {
            return;
        }

        if let Some(last_pos) = self.last_mouse_pos {
            let dx = (position.0 - last_pos.0) as f32;
            let dy = (position.1 - last_pos.1) as f32;

            let sensitivity = 0.002;
            self.camera.yaw += dx * sensitivity;
            self.camera.pitch -= dy * sensitivity;

            self.camera.pitch = self.camera.pitch.clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
        }

        self.last_mouse_pos = Some(position);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

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
