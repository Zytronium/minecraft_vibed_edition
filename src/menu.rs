/// Menu state for world size selection
pub struct MainMenu {
    pipeline: wgpu::RenderPipeline,
    white_bind_group: wgpu::BindGroup,
    dirt_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    selected_option: usize,
}

#[derive(Clone, Copy)]
pub enum WorldSize {
    Small,      // 8x8 chunks
    Medium,     // 16x16 chunks
    Large,      // 32x32 chunks
    Enormous,   // 64x64 chunks
    Overkill,   // 1024x1024 chunks
}

impl WorldSize {
    pub fn get_chunks(&self) -> i32 {
        match self {
            WorldSize::Small => 8,
            WorldSize::Medium => 16,
            WorldSize::Large => 32,
            WorldSize::Enormous => 64,
            WorldSize::Overkill => 1024,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => WorldSize::Small,
            1 => WorldSize::Medium,
            2 => WorldSize::Large,
            3 => WorldSize::Enormous,
            4 => WorldSize::Overkill,
            _ => WorldSize::Medium,
        }
    }
}

const DIRT_TEXTURE: &[u8] = include_bytes!("../assets/textures/dirt.png");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MenuVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    color: [f32; 4],
}

impl MenuVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MenuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

impl MainMenu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Load dirt texture for background
        let dirt_texture = load_texture_from_bytes(device, queue, DIRT_TEXTURE, "menu_dirt");
        let dirt_view = dirt_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create white texture for solid colors
        let white_texture = create_white_texture(device, queue);
        let white_view = white_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
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
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("menu_texture_bind_group_layout"),
        });

        let dirt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dirt_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("menu_dirt_bind_group"),
        });

        let white_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&white_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("menu_white_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Menu Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("loading.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Menu Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Menu Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[MenuVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let max_vertices = 1000;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Menu Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<MenuVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_indices = 2000;
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Menu Index Buffer"),
            size: (max_indices * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            white_bind_group,
            dirt_bind_group,
            vertex_buffer,
            index_buffer,
            selected_option: 1, // Default to Medium
        }
    }

    pub fn move_selection_up(&mut self) {
        if self.selected_option > 0 {
            self.selected_option -= 1;
        }
    }

    pub fn move_selection_down(&mut self) {
        if self.selected_option < 4 {
            self.selected_option += 1;
        }
    }

    pub fn get_selected_size(&self) -> WorldSize {
        WorldSize::from_index(self.selected_option)
    }

    pub fn update_geometry(&self, queue: &wgpu::Queue) -> usize {
        let (vertices, indices) = create_menu_geometry(self.selected_option);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        indices.len()
    }

    pub fn render(&self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue, num_indices: usize) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Menu Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Menu Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw background with dirt texture
            render_pass.set_bind_group(0, &self.dirt_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);

            // Draw UI elements with white texture
            if num_indices > 6 {
                render_pass.set_bind_group(0, &self.white_bind_group, &[]);
                render_pass.draw_indexed(6..num_indices as u32, 0, 0..1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn create_menu_geometry(selected_index: usize) -> (Vec<MenuVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Background
    let bg_verts = [
        MenuVertex { position: [-1.0, -1.0], tex_coords: [0.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        MenuVertex { position: [ 1.0, -1.0], tex_coords: [8.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        MenuVertex { position: [ 1.0,  1.0], tex_coords: [8.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        MenuVertex { position: [-1.0,  1.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
    ];
    let base = vertices.len() as u32;
    vertices.extend_from_slice(&bg_verts);
    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

    // Title: "Minecraft: Vibed Edition"
    let title = "Minecraft: Vibed Edition";
    let char_width = 0.028;
    let char_height = 0.045;
    let title_y = 0.6;
    let title_width = title.len() as f32 * char_width;
    let start_x = -title_width / 2.0;

    for (i, c) in title.chars().enumerate() {
        if c == ' ' || c == ':' {
            continue;
        }
        let x = start_x + i as f32 * char_width;
        let width = char_width * 0.8;

        let verts = [
            MenuVertex { position: [x, title_y - char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            MenuVertex { position: [x + width, title_y - char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            MenuVertex { position: [x + width, title_y + char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            MenuVertex { position: [x, title_y + char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        ];
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    // Subtitle
    let subtitle = "Select World Size";
    let subtitle_y = 0.45;
    let subtitle_width = subtitle.len() as f32 * char_width * 0.7;
    let subtitle_x = -subtitle_width / 2.0;

    for (i, c) in subtitle.chars().enumerate() {
        if c == ' ' {
            continue;
        }
        let x = subtitle_x + i as f32 * char_width * 0.7;
        let width = char_width * 0.6;

        let verts = [
            MenuVertex { position: [x, subtitle_y - char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.8, 0.8, 0.8, 1.0] },
            MenuVertex { position: [x + width, subtitle_y - char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.8, 0.8, 0.8, 1.0] },
            MenuVertex { position: [x + width, subtitle_y + char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.8, 0.8, 0.8, 1.0] },
            MenuVertex { position: [x, subtitle_y + char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.8, 0.8, 0.8, 1.0] },
        ];
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    // Menu options
    let options = [
        "Small (8x8 chunks)",
        "Medium (16x16 chunks)",
        "Large (32x32 chunks)",
        "Enormous (64x64 chunks)",
        "Overkill (1024x1024 chunks)",
    ];

    let option_start_y = 0.2;
    let option_spacing = 0.12;

    for (idx, option) in options.iter().enumerate() {
        let y = option_start_y - idx as f32 * option_spacing;
        let is_selected = idx == selected_index;

        // Selection highlight
        if is_selected {
            let highlight_width = 0.5;
            let highlight_height = 0.055;
            let highlight_verts = [
                MenuVertex { position: [-highlight_width, y - highlight_height], tex_coords: [0.0, 0.0], color: [0.3, 0.5, 0.3, 0.8] },
                MenuVertex { position: [ highlight_width, y - highlight_height], tex_coords: [0.0, 0.0], color: [0.3, 0.5, 0.3, 0.8] },
                MenuVertex { position: [ highlight_width, y + highlight_height], tex_coords: [0.0, 0.0], color: [0.3, 0.5, 0.3, 0.8] },
                MenuVertex { position: [-highlight_width, y + highlight_height], tex_coords: [0.0, 0.0], color: [0.3, 0.5, 0.3, 0.8] },
            ];
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&highlight_verts);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        // Option text
        let color = if is_selected { [1.0, 1.0, 0.0, 1.0] } else { [0.9, 0.9, 0.9, 1.0] };
        let text_width = option.len() as f32 * char_width * 0.6;
        let text_x = -text_width / 2.0;

        for (i, c) in option.chars().enumerate() {
            if c == ' ' || c == '(' || c == ')' {
                continue;
            }
            let x = text_x + i as f32 * char_width * 0.6;
            let width = char_width * 0.5;

            let verts = [
                MenuVertex { position: [x, y - char_height * 0.35], tex_coords: [0.0, 0.0], color },
                MenuVertex { position: [x + width, y - char_height * 0.35], tex_coords: [0.0, 0.0], color },
                MenuVertex { position: [x + width, y + char_height * 0.35], tex_coords: [0.0, 0.0], color },
                MenuVertex { position: [x, y + char_height * 0.35], tex_coords: [0.0, 0.0], color },
            ];
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&verts);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }
    }

    // Instructions
    let instructions = "Use arrow keys, then press ENTER";
    let instr_y = -0.6;
    let instr_width = instructions.len() as f32 * char_width * 0.5;
    let instr_x = -instr_width / 2.0;

    for (i, c) in instructions.chars().enumerate() {
        if c == ' ' || c == ',' {
            continue;
        }
        let x = instr_x + i as f32 * char_width * 0.5;
        let width = char_width * 0.4;

        let verts = [
            MenuVertex { position: [x, instr_y - char_height * 0.3], tex_coords: [0.0, 0.0], color: [0.6, 0.6, 0.6, 1.0] },
            MenuVertex { position: [x + width, instr_y - char_height * 0.3], tex_coords: [0.0, 0.0], color: [0.6, 0.6, 0.6, 1.0] },
            MenuVertex { position: [x + width, instr_y + char_height * 0.3], tex_coords: [0.0, 0.0], color: [0.6, 0.6, 0.6, 1.0] },
            MenuVertex { position: [x, instr_y + char_height * 0.3], tex_coords: [0.0, 0.0], color: [0.6, 0.6, 0.6, 1.0] },
        ];
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    (vertices, indices)
}

fn create_white_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let data = [255u8, 255, 255, 255];
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Menu White Texture"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
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
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );

    texture
}

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
