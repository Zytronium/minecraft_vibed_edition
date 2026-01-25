use wgpu::util::DeviceExt;

/// Simple vertex for 2D quad rendering (for loading screen)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    color: [f32; 4],
}

impl QuadVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
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

/// Loading screen renderer with text support
pub struct LoadingScreen {
    pipeline: wgpu::RenderPipeline,
    text_pipeline: wgpu::RenderPipeline,
    texture_bind_group: wgpu::BindGroup,
    font_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    text_vertex_buffer: wgpu::Buffer,
    text_index_buffer: wgpu::Buffer,
    num_indices: u32,
    text_num_indices: u32,
}

impl LoadingScreen {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Load dirt texture for background
        let dirt_texture = load_texture(device, queue, "assets/textures/dirt.png");
        let dirt_view = dirt_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create simple bitmap font texture (8x8 ASCII characters)
        let font_texture = create_font_texture(device, queue);
        let font_view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            label: Some("loading_texture_bind_group_layout"),
        });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            label: Some("loading_texture_bind_group"),
        });

        let font_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&font_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("font_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Loading Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("loading.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Loading Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Loading Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[QuadVertex::desc()],
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

        let text_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Text Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[QuadVertex::desc()],
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

        // Create initial geometry (will be updated with progress)
        let (vertices, indices) = create_loading_geometry(0.0);

        let max_vertices = 16;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loading Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Loading Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create text geometry for "Generating World..."
        let (text_vertices, text_indices) = create_text_geometry("Generating World...", 0.0, 0.2);

        let text_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Text Vertex Buffer"),
            contents: bytemuck::cast_slice(&text_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let text_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Text Index Buffer"),
            contents: bytemuck::cast_slice(&text_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            pipeline,
            text_pipeline,
            texture_bind_group,
            font_bind_group,
            vertex_buffer,
            index_buffer,
            text_vertex_buffer,
            text_index_buffer,
            num_indices: indices.len() as u32,
            text_num_indices: text_indices.len() as u32,
        }
    }

    /// Update the progress bar (progress from 0.0 to 1.0)
    pub fn update_progress(&self, queue: &wgpu::Queue, progress: f32) {
        let (vertices, _) = create_loading_geometry(progress);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    }

    /// Render the loading screen
    pub fn render(&self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Loading Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Loading Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw background and progress bar
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            // Draw text
            render_pass.set_pipeline(&self.text_pipeline);
            render_pass.set_bind_group(0, &self.font_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.text_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.text_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.text_num_indices, 0, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Create geometry for the loading screen with a progress bar
fn create_loading_geometry(progress: f32) -> (Vec<QuadVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Background (tiled dirt texture)
    let bg_verts = [
        QuadVertex { position: [-1.0, -1.0], tex_coords: [0.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [ 1.0, -1.0], tex_coords: [8.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [ 1.0,  1.0], tex_coords: [8.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [-1.0,  1.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
    ];

    let base = vertices.len() as u32;
    vertices.extend_from_slice(&bg_verts);
    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

    // Progress bar background (dark gray with border)
    let bar_width = 0.6;
    let bar_height = 0.04;
    let bar_y = -0.1;

    let bar_bg_verts = [
        QuadVertex { position: [-bar_width, bar_y - bar_height], tex_coords: [0.0, 1.0], color: [0.2, 0.2, 0.2, 0.95] },
        QuadVertex { position: [ bar_width, bar_y - bar_height], tex_coords: [1.0, 1.0], color: [0.2, 0.2, 0.2, 0.95] },
        QuadVertex { position: [ bar_width, bar_y + bar_height], tex_coords: [1.0, 0.0], color: [0.2, 0.2, 0.2, 0.95] },
        QuadVertex { position: [-bar_width, bar_y + bar_height], tex_coords: [0.0, 0.0], color: [0.2, 0.2, 0.2, 0.95] },
    ];

    let base = vertices.len() as u32;
    vertices.extend_from_slice(&bar_bg_verts);
    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

    // Progress bar fill (bright green)
    let padding = 0.005;
    if progress > 0.01 {
        let fill_width = (bar_width - padding) * progress.min(1.0);

        let bar_fill_verts = [
            QuadVertex { position: [-bar_width + padding, bar_y - bar_height + padding], tex_coords: [0.0, 1.0], color: [0.2, 0.9, 0.2, 1.0] },
            QuadVertex { position: [-bar_width + padding + fill_width, bar_y - bar_height + padding], tex_coords: [1.0, 1.0], color: [0.2, 0.9, 0.2, 1.0] },
            QuadVertex { position: [-bar_width + padding + fill_width, bar_y + bar_height - padding], tex_coords: [1.0, 0.0], color: [0.2, 0.9, 0.2, 1.0] },
            QuadVertex { position: [-bar_width + padding, bar_y + bar_height - padding], tex_coords: [0.0, 0.0], color: [0.2, 0.9, 0.2, 1.0] },
        ];

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&bar_fill_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    (vertices, indices)
}

/// Create text geometry using bitmap font
fn create_text_geometry(text: &str, x: f32, y: f32) -> (Vec<QuadVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let char_width = 0.04;
    let char_height = 0.06;
    let text_width = text.len() as f32 * char_width;
    let start_x = x - text_width / 2.0;

    for (i, c) in text.chars().enumerate() {
        let char_x = start_x + i as f32 * char_width;

        // Calculate texture coordinates for this character in the font atlas
        let char_index = c as u32;
        let chars_per_row = 16;
        let tx = (char_index % chars_per_row) as f32 / chars_per_row as f32;
        let ty = (char_index / chars_per_row) as f32 / 16.0;
        let tw = 1.0 / chars_per_row as f32;
        let th = 1.0 / 16.0;

        let char_verts = [
            QuadVertex { position: [char_x, y - char_height], tex_coords: [tx, ty + th], color: [1.0, 1.0, 1.0, 1.0] },
            QuadVertex { position: [char_x + char_width, y - char_height], tex_coords: [tx + tw, ty + th], color: [1.0, 1.0, 1.0, 1.0] },
            QuadVertex { position: [char_x + char_width, y + char_height], tex_coords: [tx + tw, ty], color: [1.0, 1.0, 1.0, 1.0] },
            QuadVertex { position: [char_x, y + char_height], tex_coords: [tx, ty], color: [1.0, 1.0, 1.0, 1.0] },
        ];

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&char_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    (vertices, indices)
}

/// Create a simple bitmap font texture (8x8 ASCII grid)
fn create_font_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    // Create a 256x256 texture (16x16 grid of 16x16 characters)
    let size = 256;
    let char_size = 16;
    let mut data = vec![0u8; size * size * 4];

    // Simple pixel font data for basic characters
    for c in 32u8..127u8 {
        let char_x = ((c % 16) as usize) * char_size;
        let char_y = ((c / 16) as usize) * char_size;

        // Draw simple character patterns
        draw_char(&mut data, size, char_x, char_y, c);
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Font Texture"),
        size: wgpu::Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        },
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
            bytes_per_row: Some(4 * size as u32),
            rows_per_image: Some(size as u32),
        },
        wgpu::Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        },
    );

    texture
}

/// Draw a character into the font texture
fn draw_char(data: &mut [u8], width: usize, x: usize, y: usize, c: u8) {
    let set_pixel = |data: &mut [u8], px: usize, py: usize| {
        if px < width && py < width {
            let idx = (py * width + px) * 4;
            data[idx] = 255;     // R
            data[idx + 1] = 255; // G
            data[idx + 2] = 255; // B
            data[idx + 3] = 255; // A
        }
    };

    // Simple 5x7 pixel patterns for readable characters
    match c {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b' ' | b'.' | b',' | b'!' | b'?' => {
            // Draw vertical line for letters
            for dy in 2..14 {
                set_pixel(data, x + 4, y + dy);
                set_pixel(data, x + 12, y + dy);
            }
            // Draw horizontal lines
            set_pixel(data, x + 5, y + 2);
            set_pixel(data, x + 6, y + 2);
            set_pixel(data, x + 7, y + 2);
            set_pixel(data, x + 8, y + 2);
            set_pixel(data, x + 9, y + 2);
            set_pixel(data, x + 10, y + 2);
            set_pixel(data, x + 11, y + 2);

            set_pixel(data, x + 5, y + 8);
            set_pixel(data, x + 6, y + 8);
            set_pixel(data, x + 7, y + 8);
            set_pixel(data, x + 8, y + 8);
            set_pixel(data, x + 9, y + 8);
            set_pixel(data, x + 10, y + 8);
            set_pixel(data, x + 11, y + 8);
        }
        _ => {}
    }
}

/// Load a texture from disk
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
