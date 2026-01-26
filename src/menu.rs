use glyphon::{
    Attrs, Buffer, Color as GlyphonColor, Family, FontSystem, Metrics,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

const FONT: &[u8] = include_bytes!("../assets/fonts/font.ttf");
const DIRT_TEXTURE: &[u8] = include_bytes!("../assets/textures/dirt.png");

pub struct MainMenu {
    pipeline: wgpu::RenderPipeline,
    dirt_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    selected_option: usize,

    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    buffers: Vec<Buffer>,
}

#[derive(Clone, Copy, Debug)]
pub enum WorldSize {
    Small,
    Medium,
    Large,
    Enormous,
    Overkill,
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
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let dirt_texture = load_texture_from_bytes(device, queue, DIRT_TEXTURE, "menu_dirt");
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

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Menu Vertex Buffer"),
            size: (1000 * std::mem::size_of::<MenuVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Menu Index Buffer"),
            size: (2000 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize text rendering
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(FONT.to_vec());

        let swash_cache = SwashCache::new();
        let cache = glyphon::Cache::new(device);
        let mut viewport = Viewport::new(device, &cache);
        // CRITICAL FIX: Update viewport with actual window dimensions
        viewport.update(queue, glyphon::Resolution { width, height });

        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        // Use the loaded font by specifying Family::Name with the font family name
        // Most fonts will work with their family name, but we can also use Family::SansSerif as fallback
        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));

        // Create text buffers
        let mut buffers = Vec::new();

        // Title
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(60.0, 70.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "Minecraft: Vibed Edition", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        // Subtitle
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(40.0, 50.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "Select World Size", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        // Options
        let options = [
            "Small (8x8 chunks)",
            "Medium (16x16 chunks)",
            "Large (32x32 chunks)",
            "Enormous (64x64 chunks)",
            "Overkill (1024x1024 chunks)",
        ];
        for option in options {
            let mut buffer = Buffer::new(&mut font_system, Metrics::new(35.0, 45.0));
            buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
            buffer.set_text(&mut font_system, option, font_attrs, Shaping::Advanced);
            buffers.push(buffer);
        }

        // Instructions
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(25.0, 35.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "Use arrow keys, then press ENTER", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        Self {
            pipeline,
            dirt_bind_group,
            vertex_buffer,
            index_buffer,
            selected_option: 1,
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            buffers,
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
        let bg_verts = [
            MenuVertex { position: [-1.0, -1.0], tex_coords: [0.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
            MenuVertex { position: [ 1.0, -1.0], tex_coords: [8.0, 8.0], color: [1.0, 1.0, 1.0, 1.0] },
            MenuVertex { position: [ 1.0,  1.0], tex_coords: [8.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
            MenuVertex { position: [-1.0,  1.0], tex_coords: [0.0, 0.0], color: [1.0, 1.0, 1.0, 1.0] },
        ];
        let indices = [0u32, 1, 2, 2, 3, 0];

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&bg_verts));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        6
    }

    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue, _num_indices: usize) {
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
            render_pass.set_bind_group(0, &self.dirt_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

        // Prepare text areas
        let mut text_areas = vec![
            TextArea {
                buffer: &self.buffers[0],
                left: 340.0,
                top: 100.0,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: 1280, bottom: 720 },
                default_color: GlyphonColor::rgb(255, 255, 255),
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.buffers[1],
                left: 440.0,
                top: 200.0,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: 1280, bottom: 720 },
                default_color: GlyphonColor::rgb(204, 204, 204),
                custom_glyphs: &[],
            },
        ];

        // Options
        for i in 0..5 {
            let y = 300.0 + i as f32 * 60.0;
            let color = if i == self.selected_option {
                GlyphonColor::rgb(255, 255, 0)
            } else {
                GlyphonColor::rgb(230, 230, 230)
            };

            text_areas.push(TextArea {
                buffer: &self.buffers[2 + i],
                left: 480.0,
                top: y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: 1280, bottom: 720 },
                default_color: color,
                custom_glyphs: &[],
            });
        }

        // Instructions
        text_areas.push(TextArea {
            buffer: &self.buffers[7],
            left: 380.0,
            top: 650.0,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: 1280, bottom: 720 },
            default_color: GlyphonColor::rgb(153, 153, 153),
            custom_glyphs: &[],
        });

        // Prepare text areas
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
