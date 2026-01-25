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

/// Loading screen renderer
pub struct LoadingScreen {
    pipeline: wgpu::RenderPipeline,
    dirt_bind_group: wgpu::BindGroup,
    white_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl LoadingScreen {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Load dirt texture for background
        let dirt_texture = load_texture(device, queue, "assets/textures/dirt.png");
        let dirt_view = dirt_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create white 1x1 texture for solid color rendering
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
            label: Some("loading_texture_bind_group_layout"),
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
            label: Some("dirt_bind_group"),
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
            label: Some("white_bind_group"),
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

        // Create buffers - large enough for all geometry
        // Max: background(4) + "Generating World..."(~18*4=72) + progress text(~7*4=28) + bar(8) = ~112 vertices
        let max_vertices = 150;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loading Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_indices = 300;
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loading Index Buffer"),
            size: (max_indices * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            dirt_bind_group,
            white_bind_group,
            vertex_buffer,
            index_buffer,
        }
    }

    /// Update the progress bar (progress from 0.0 to 1.0, current chunk number)
    pub fn update_progress(&self, queue: &wgpu::Queue, progress: f32, chunk_num: usize) -> usize {
        let (vertices, indices) = create_loading_geometry(progress, chunk_num);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        indices.len()
    }

    /// Render the loading screen
    pub fn render(&self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue, num_indices: usize) {
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

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Draw background with dirt texture (first 6 indices)
            render_pass.set_bind_group(0, &self.dirt_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);

            // Draw everything else with white texture (for solid colors)
            if num_indices > 6 {
                render_pass.set_bind_group(0, &self.white_bind_group, &[]);
                render_pass.draw_indexed(6..num_indices as u32, 0, 0..1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Create geometry for the loading screen with progress bar and text
fn create_loading_geometry(progress: f32, chunk_num: usize) -> (Vec<QuadVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Background (tiled dirt texture) - FIRST so it uses dirt bind group
    let bg_verts = [
        QuadVertex { position: [-1.0, -1.0], tex_coords: [0.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [ 1.0, -1.0], tex_coords: [8.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [ 1.0,  1.0], tex_coords: [8.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [-1.0,  1.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
    ];

    let base = vertices.len() as u32;
    vertices.extend_from_slice(&bg_verts);
    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

    // ALL GEOMETRY BELOW THIS USES WHITE TEXTURE (solid colors via vertex color)

    // "Generating World..." text (simple block letters)
    let text = "Generating World...";
    let char_width = 0.035;
    let char_height = 0.05;
    let text_y = 0.15;
    let text_width = text.len() as f32 * char_width;
    let start_x = -text_width / 2.0;

    for (i, c) in text.chars().enumerate() {
        if c == ' ' {
            continue;
        }

        let x = start_x + i as f32 * char_width;
        let width = if c == '.' { char_width * 0.3 } else { char_width * 0.8 };

        // White rectangles for letters
        let char_verts = [
            QuadVertex { position: [x, text_y - char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            QuadVertex { position: [x + width, text_y - char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            QuadVertex { position: [x + width, text_y + char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            QuadVertex { position: [x, text_y + char_height / 2.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        ];

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&char_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    // Progress text (e.g., "128/256")
    let progress_text = format!("{}/256", chunk_num);
    let progress_y = 0.03;
    let progress_text_width = progress_text.len() as f32 * char_width * 0.6;
    let progress_start_x = -progress_text_width / 2.0;

    for (i, c) in progress_text.chars().enumerate() {
        if c == ' ' {
            continue;
        }

        let x = progress_start_x + i as f32 * char_width * 0.6;
        let width = if c == '/' { char_width * 0.4 } else { char_width * 0.5 };

        let char_verts = [
            QuadVertex { position: [x, progress_y - char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.9, 0.9, 0.9, 1.0] },
            QuadVertex { position: [x + width, progress_y - char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.9, 0.9, 0.9, 1.0] },
            QuadVertex { position: [x + width, progress_y + char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.9, 0.9, 0.9, 1.0] },
            QuadVertex { position: [x, progress_y + char_height * 0.35], tex_coords: [0.0, 0.0], color: [0.9, 0.9, 0.9, 1.0] },
        ];

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&char_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    // Progress bar background (dark gray)
    let bar_width = 0.6;
    let bar_height = 0.04;
    let bar_y = -0.1;

    let bar_bg_verts = [
        QuadVertex { position: [-bar_width, bar_y - bar_height], tex_coords: [0.0, 0.0], color: [0.2, 0.2, 0.2, 0.95] },
        QuadVertex { position: [ bar_width, bar_y - bar_height], tex_coords: [0.0, 0.0], color: [0.2, 0.2, 0.2, 0.95] },
        QuadVertex { position: [ bar_width, bar_y + bar_height], tex_coords: [0.0, 0.0], color: [0.2, 0.2, 0.2, 0.95] },
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
            QuadVertex { position: [-bar_width + padding, bar_y - bar_height + padding], tex_coords: [0.0, 0.0], color: [0.2, 0.9, 0.2, 1.0] },
            QuadVertex { position: [-bar_width + padding + fill_width, bar_y - bar_height + padding], tex_coords: [0.0, 0.0], color: [0.2, 0.9, 0.2, 1.0] },
            QuadVertex { position: [-bar_width + padding + fill_width, bar_y + bar_height - padding], tex_coords: [0.0, 0.0], color: [0.2, 0.9, 0.2, 1.0] },
            QuadVertex { position: [-bar_width + padding, bar_y + bar_height - padding], tex_coords: [0.0, 0.0], color: [0.2, 0.9, 0.2, 1.0] },
        ];

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&bar_fill_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    (vertices, indices)
}

/// Create a 1x1 white texture for solid color rendering
fn create_white_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let data = [255u8, 255, 255, 255]; // White pixel

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("White Texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
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
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    texture
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
