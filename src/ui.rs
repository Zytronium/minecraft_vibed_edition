/// UI rendering system for crosshair and hotbar
use crate::block;

/// Simple 2D vertex for UI elements (screen space coordinates)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UIVertex {
    pub position: [f32; 2],      // Screen space position (0,0 = top-left, 1,1 = bottom-right)
    pub tex_coords: [f32; 2],    // UV coordinates for texture
    pub color: [f32; 4],         // RGBA color (for crosshair)
    pub texture_index: u32,      // Which texture to use (0 = none/color only)
}

impl UIVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UIVertex>() as wgpu::BufferAddress,
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
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() * 2 + std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// Hotbar state - tracks which block is selected
pub struct Hotbar {
    pub selected_slot: usize,  // 0-8
    pub blocks: [block::BlockType; 9],  // The 9 blocks in the hotbar
    pub block_textures: [&'static str; 9],  // Texture names for icons (top face of each block)
}

impl Hotbar {
    pub fn new(registry: &block::BlockRegistry) -> Self {
        Self {
            selected_slot: 0,
            blocks: [
                registry.block_type("minecraft:grass"),
                registry.block_type("minecraft:dirt"),
                registry.block_type("minecraft:stone"),
                registry.block_type("minecraft:cobblestone"),
                registry.block_type("minecraft:planks"),
                registry.block_type("minecraft:log"),
                registry.block_type("minecraft:leaves"),
                registry.block_type("minecraft:glass"),
                registry.block_type("minecraft:bedrock"),
            ],
            block_textures: [
                "grass_top",      // Grass
                "dirt",           // Dirt
                "stone",          // Stone
                "cobblestone",    // Cobblestone
                "planks",         // Planks
                "log_top-bottom", // Log
                "leaves",         // Leaves
                "glass",          // Glass
                "bedrock",        // Bedrock
            ],
        }
    }

    pub fn select_slot(&mut self, slot: usize) {
        if slot < 9 {
            self.selected_slot = slot;
        }
    }

    pub fn scroll(&mut self, delta: i32) {
        let new_slot = (self.selected_slot as i32 + delta).rem_euclid(9) as usize;
        self.selected_slot = new_slot;
    }

    pub fn get_selected_block(&self) -> block::BlockType {
        self.blocks[self.selected_slot]
    }
}

/// UI Renderer - handles crosshair and hotbar rendering
pub struct UIRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    crosshair_num_indices: u32,  // Separate count for crosshair
    pipeline: wgpu::RenderPipeline,
    crosshair_pipeline: wgpu::RenderPipeline,  // Separate pipeline for inversion
    texture_bind_group: wgpu::BindGroup,
    screen_width: u32,
    screen_height: u32,
    _dummy_texture: wgpu::Texture,         // Dummy texture for index 0
    _slot_texture: wgpu::Texture,           // Keep textures alive
    _slot_selected_texture: wgpu::Texture,  // Keep textures alive
    _dummy_view: wgpu::TextureView,         // Keep views alive
    _slot_view: wgpu::TextureView,          // Keep views alive
    _slot_selected_view: wgpu::TextureView, // Keep views alive
}

impl UIRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        screen_width: u32,
        screen_height: u32,
        hotbar_slot_texture: &[u8],
        hotbar_slot_selected_texture: &[u8],
        _block_textures: &std::collections::HashMap<String, u32>,
        block_texture_views: &[&wgpu::TextureView],
    ) -> Self {
        // Load UI textures
        let slot_texture = Self::load_texture(device, queue, hotbar_slot_texture, "hotbar_slot");
        let slot_selected_texture = Self::load_texture(device, queue, hotbar_slot_selected_texture, "hotbar_slot_selected");

        // Create a 1x1 dummy white texture for index 0 (reserved for "no texture" in shader)
        let dummy_texture = Self::create_dummy_texture(device, queue);
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create views for UI textures
        let slot_view = slot_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let slot_selected_view = slot_selected_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Build texture array with proper indices:
        // Index 0: Dummy (for "no texture" mode in shader)
        // Index 1: Normal hotbar slot texture
        // Index 2: Selected hotbar slot texture
        // Index 3+: Block textures (in same order as world renderer)
        let mut all_texture_views: Vec<&wgpu::TextureView> = vec![
            &dummy_view,         // Index 0 - dummy for "no texture" mode
            &slot_view,          // Index 1 - normal hotbar slot
            &slot_selected_view, // Index 2 - selected hotbar slot
        ];
        all_texture_views.extend(block_texture_views.iter().copied()); // Index 3+ - block textures

        // Create sampler
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
        let texture_count = all_texture_views.len() as u32;
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
            label: Some("ui_texture_bind_group_layout"),
        });

        // Create bind group
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&all_texture_views),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("ui_texture_bind_group"),
        });

        // Load UI shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
        });

        // Create render pipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Normal UI pipeline with alpha blending
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[UIVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,  // No culling for UI
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,  // UI doesn't need depth testing
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Crosshair pipeline with color inversion blend mode
        let crosshair_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Crosshair Inversion Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[UIVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDst,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        // Create empty buffers (will be updated each frame)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Vertex Buffer"),
            size: 4096,  // Enough for UI elements
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Index Buffer"),
            size: 4096,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: 0,
            crosshair_num_indices: 0,
            pipeline,
            crosshair_pipeline,
            texture_bind_group,
            screen_width,
            screen_height,
            _dummy_texture: dummy_texture,
            _slot_texture: slot_texture,
            _slot_selected_texture: slot_selected_texture,
            _dummy_view: dummy_view,
            _slot_view: slot_view,
            _slot_selected_view: slot_selected_view,
        }
    }

    fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        // Create a 1x1 white texture as a placeholder for index 0
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // White pixel
        let pixel_data: [u8; 4] = [255, 255, 255, 255];
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixel_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            size,
        );

        texture
    }

    fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], label: &str) -> wgpu::Texture {
        let img = image::load_from_memory(bytes)
            .unwrap_or_else(|e| panic!("Failed to load UI texture {}: {}", label, e))
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

    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    pub fn update_geometry(
        &mut self,
        queue: &wgpu::Queue,
        hotbar: &Hotbar,
        block_textures: &std::collections::HashMap<String, u32>,
    ) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Add crosshair (inverted colors, maintains 1:1 aspect ratio)
        let _aspect = self.screen_width as f32 / self.screen_height as f32;
        let crosshair_size_pixels = 8.0;  // Size in pixels
        let crosshair_thickness_pixels = 2.0;

        // Convert pixels to normalized coordinates, accounting for aspect ratio
        let crosshair_size_x = crosshair_size_pixels / self.screen_width as f32;
        let crosshair_size_y = crosshair_size_pixels / self.screen_height as f32;
        let crosshair_thickness_x = crosshair_thickness_pixels / self.screen_width as f32;
        let crosshair_thickness_y = crosshair_thickness_pixels / self.screen_height as f32;

        // Use special color value to signal inversion in shader (negative alpha)
        let inverted = [1.0, 1.0, 1.0, -1.0];  // Negative alpha = invert mode

        // Horizontal line
        self.add_rect(
            &mut vertices,
            &mut indices,
            0.5 - crosshair_size_x, 0.5 - crosshair_thickness_y,
            0.5 + crosshair_size_x, 0.5 + crosshair_thickness_y,
            inverted,
            0, // No texture, just color
        );

        // Vertical line
        self.add_rect(
            &mut vertices,
            &mut indices,
            0.5 - crosshair_thickness_x, 0.5 - crosshair_size_y,
            0.5 + crosshair_thickness_x, 0.5 + crosshair_size_y,
            inverted,
            0, // No texture, just color
        );

        // Store crosshair index count (12 indices = 2 rects * 6 indices per rect)
        self.crosshair_num_indices = indices.len() as u32;

        // Add hotbar at bottom of screen
        let slot_size = 50.0;  // Size in pixels
        let slot_spacing = 4.0;  // Spacing between slots in pixels
        let total_width = 9.0 * slot_size + 8.0 * slot_spacing;
        let start_x = (self.screen_width as f32 - total_width) / 2.0;
        let start_y = self.screen_height as f32 - slot_size - 20.0;

        for i in 0..9 {
            let x = start_x + i as f32 * (slot_size + slot_spacing);
            let y = start_y;

            // Convert to normalized coordinates
            let x1 = x / self.screen_width as f32;
            let y1 = y / self.screen_height as f32;
            let x2 = (x + slot_size) / self.screen_width as f32;
            let y2 = (y + slot_size) / self.screen_height as f32;

            // Draw slot background
            let slot_texture_index = if i == hotbar.selected_slot { 2 } else { 1 };
            self.add_rect(
                &mut vertices,
                &mut indices,
                x1, y1, x2, y2,
                [1.0, 1.0, 1.0, 1.0],
                slot_texture_index,
            );

            // Draw block icon inside slot (slightly smaller)
            let icon_padding = 6.0;
            let icon_x1 = (x + icon_padding) / self.screen_width as f32;
            let icon_y1 = (y + icon_padding) / self.screen_height as f32;
            let icon_x2 = (x + slot_size - icon_padding) / self.screen_width as f32;
            let icon_y2 = (y + slot_size - icon_padding) / self.screen_height as f32;

            // Get the block's texture name from hotbar
            let texture_name = hotbar.block_textures[i];

            if let Some(&texture_index) = block_textures.get(texture_name) {
                // Add 3 to skip dummy + 2 UI slot textures
                self.add_rect(
                    &mut vertices,
                    &mut indices,
                    icon_x1, icon_y1, icon_x2, icon_y2,
                    [1.0, 1.0, 1.0, 1.0],
                    texture_index + 3,  // Offset by 3 for dummy + 2 UI textures
                );
            }
        }

        // Upload to GPU
        self.num_indices = indices.len() as u32;
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
    }

    fn add_rect(
        &self,
        vertices: &mut Vec<UIVertex>,
        indices: &mut Vec<u32>,
        x1: f32, y1: f32, x2: f32, y2: f32,
        color: [f32; 4],
        texture_index: u32,
    ) {
        let base_index = vertices.len() as u32;

        // Add 4 vertices for the rectangle
        vertices.push(UIVertex {
            position: [x1, y1],
            tex_coords: [0.0, 0.0],
            color,
            texture_index,
        });
        vertices.push(UIVertex {
            position: [x2, y1],
            tex_coords: [1.0, 0.0],
            color,
            texture_index,
        });
        vertices.push(UIVertex {
            position: [x2, y2],
            tex_coords: [1.0, 1.0],
            color,
            texture_index,
        });
        vertices.push(UIVertex {
            position: [x1, y2],
            tex_coords: [0.0, 1.0],
            color,
            texture_index,
        });

        // Add 6 indices for 2 triangles
        indices.extend_from_slice(&[
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3,
        ]);
    }

    pub fn render(
        &self,
        render_pass: &mut wgpu::RenderPass,
    ) {
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        // Render crosshair with inversion blend mode
        if self.crosshair_num_indices > 0 {
            render_pass.set_pipeline(&self.crosshair_pipeline);
            render_pass.draw_indexed(0..self.crosshair_num_indices, 0, 0..1);
        }

        // Render rest of UI with normal alpha blending
        if self.num_indices > self.crosshair_num_indices {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.draw_indexed(self.crosshair_num_indices..self.num_indices, 0, 0..1);
        }
    }
}
