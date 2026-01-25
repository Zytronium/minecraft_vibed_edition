mod block;
mod camera;
mod vertex;
mod world;

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
use world::World;

/// Main rendering state - holds all GPU resources and game state
struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline_opaque: wgpu::RenderPipeline,
    render_pipeline_transparent: wgpu::RenderPipeline,
    opaque_vertex_buffer: wgpu::Buffer,
    opaque_index_buffer: wgpu::Buffer,
    opaque_num_indices: u32,
    transparent_vertex_buffer: wgpu::Buffer,
    transparent_index_buffer: wgpu::Buffer,
    transparent_num_indices: u32,
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

        // Load all block textures - ORDER MATTERS! Must match BlockType::get_texture_indices()
        let texture_paths = [
            "assets/textures/grass_top.png",
            "assets/textures/grass_bottom.png",
            "assets/textures/grass_side.png",
            "assets/textures/dirt.png",
            "assets/textures/stone.png",
            "assets/textures/log_top-bottom.png",
            "assets/textures/log_side.png",
            "assets/textures/leaves.png",
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
                depth_write_enabled: true,  // Write depth for opaque
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),  // Alpha blending for transparency
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

        // Generate world with all chunks and trees
        let (all_opaque_vertices, all_opaque_indices, all_transparent_vertices, all_transparent_indices) =
            World::generate();

        println!("Generated {} opaque vertices, {} transparent vertices",
                 all_opaque_vertices.len(), all_transparent_vertices.len());

        // Upload opaque mesh to GPU
        let opaque_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Opaque Vertex Buffer"),
            contents: bytemuck::cast_slice(&all_opaque_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let opaque_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Opaque Index Buffer"),
            contents: bytemuck::cast_slice(&all_opaque_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Upload transparent mesh to GPU
        let transparent_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transparent Vertex Buffer"),
            contents: bytemuck::cast_slice(&all_transparent_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let transparent_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transparent Index Buffer"),
            contents: bytemuck::cast_slice(&all_transparent_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline_opaque,
            render_pipeline_transparent,
            opaque_vertex_buffer,
            opaque_index_buffer,
            opaque_num_indices: all_opaque_indices.len() as u32,
            transparent_vertex_buffer,
            transparent_index_buffer,
            transparent_num_indices: all_transparent_indices.len() as u32,
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

            // Draw opaque geometry first (with depth writes)
            render_pass.set_pipeline(&self.render_pipeline_opaque);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.opaque_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.opaque_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.opaque_num_indices, 0, 0..1);

            // Draw transparent geometry second (without depth writes, with blending)
            render_pass.set_pipeline(&self.render_pipeline_transparent);
            render_pass.set_vertex_buffer(0, self.transparent_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.transparent_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.transparent_num_indices, 0, 0..1);
        }

        // Submit commands and present frame
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Load a texture from disk and upload to GPU
fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> wgpu::Texture {
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

/// Application state for winit event loop
struct App {
    state: Option<State>,
    last_render_time: std::time::Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Minecraft: Fixed Transparency")
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