use glyphon::{
    Attrs, Buffer, Color as GlyphonColor, Family, FontSystem, Metrics,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

// Embed assets at compile time
const DIRT_TEXTURE: &[u8] = include_bytes!("../assets/textures/dirt.png");
const FONT: &[u8] = include_bytes!("../assets/fonts/font.ttf");

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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    // Text rendering
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    title_buffer: Buffer,
    progress_buffer: Buffer,
}

impl LoadingScreen {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Load dirt texture from embedded bytes for background
        let dirt_texture = load_texture_from_bytes(device, queue, DIRT_TEXTURE, "dirt");
        let dirt_view = dirt_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        // Create buffers for progress bar
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loading Vertex Buffer"),
            size: (100 * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loading Index Buffer"),
            size: (200 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize text rendering
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(FONT.to_vec());

        let swash_cache = SwashCache::new();
        let cache = glyphon::Cache::new(device);
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(queue, glyphon::Resolution { width: 1280, height: 720 });

        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));

        // Title text
        let mut title_buffer = Buffer::new(&mut font_system, Metrics::new(50.0, 60.0));
        title_buffer.set_size(&mut font_system, Some(1280.0), Some(720.0));
        title_buffer.set_text(&mut font_system, "Generating World...", font_attrs, Shaping::Advanced);

        // Progress text (will be updated)
        let mut progress_buffer = Buffer::new(&mut font_system, Metrics::new(30.0, 40.0));
        progress_buffer.set_size(&mut font_system, Some(1280.0), Some(720.0));
        progress_buffer.set_text(&mut font_system, "0/0", font_attrs, Shaping::Advanced);

        Self {
            pipeline,
            dirt_bind_group,
            vertex_buffer,
            index_buffer,
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            title_buffer,
            progress_buffer,
        }
    }

    /// Update the title text (for switching between "Generating World" and "Loading World")
    pub fn set_title(&mut self, title: &str) {
        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));
        self.title_buffer.set_text(&mut self.font_system, title, font_attrs, Shaping::Advanced);
    }

    /// Update the progress bar (progress from 0.0 to 1.0, current chunk number, total chunks)
    pub fn update_progress(&mut self, queue: &wgpu::Queue, progress: f32, chunk_num: usize, total_chunks: usize) -> usize {
        // Update progress text
        let progress_text = format!("{}/{}", chunk_num, total_chunks);
        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));
        self.progress_buffer.set_text(&mut self.font_system, &progress_text, font_attrs, Shaping::Advanced);

        let (vertices, indices) = create_progress_bar(progress);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        indices.len()
    }

    /// Render the loading screen
    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue, num_indices: usize) {
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

            // Draw progress bar
            if num_indices > 6 {
                render_pass.draw_indexed(6..num_indices as u32, 0, 0..1);
            }
        }

        // Render text
        let text_areas = vec![
            TextArea {
                buffer: &self.title_buffer,
                left: 450.0,
                top: 250.0,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: 1280, bottom: 720 },
                default_color: GlyphonColor::rgb(255, 255, 255),
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.progress_buffer,
                left: 580.0,
                top: 400.0,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: 1280, bottom: 720 },
                default_color: GlyphonColor::rgb(200, 200, 200),
                custom_glyphs: &[],
            },
        ];

        self.text_renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.text_renderer.render(&self.atlas, &self.viewport, &mut pass).unwrap();
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Create geometry for the progress bar only
fn create_progress_bar(progress: f32) -> (Vec<QuadVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Background (tiled dirt texture) - FIRST
    let bg_verts = [
        QuadVertex { position: [-1.0, -1.0], tex_coords: [0.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [ 1.0, -1.0], tex_coords: [8.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [ 1.0,  1.0], tex_coords: [8.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        QuadVertex { position: [-1.0,  1.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
    ];

    let base = vertices.len() as u32;
    vertices.extend_from_slice(&bg_verts);
    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

    // Progress bar background (dark gray)
    let bar_width = 0.6;
    let bar_height = 0.04;
    let bar_y = -0.3;

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

/// Load a texture from embedded bytes
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
