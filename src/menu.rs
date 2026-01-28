use glyphon::{
    Attrs, Buffer, Color as GlyphonColor, Family, FontSystem, Metrics,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

const FONT: &[u8] = include_bytes!("../assets/fonts/font.ttf");
const BUTTON_TEXTURE: &[u8] = include_bytes!("../assets/textures/ui/button2.png");
const WALLPAPER: &[u8] = include_bytes!("../assets/wallpaper.png");

pub struct MainMenu {
    pipeline: wgpu::RenderPipeline,
    wallpaper_bind_group: wgpu::BindGroup,
    button_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    hovered_option: Option<usize>,
    window_width: u32,
    window_height: u32,
    wallpaper_width: u32,
    wallpaper_height: u32,

    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    buffers: Vec<Buffer>,

    // World naming fields
    world_name_input: String,
    world_name_error: Option<String>,
    input_focused: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum WorldSize {
    Tiny,
    Small,
    Medium,
    Large,
    Enormous,
}

impl WorldSize {
    pub fn get_chunks(&self) -> i32 {
        match self {
            WorldSize::Tiny => 8,
            WorldSize::Small => 16,
            WorldSize::Medium => 24,
            WorldSize::Large => 32,
            WorldSize::Enormous => 64,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => WorldSize::Tiny,
            1 => WorldSize::Small,
            2 => WorldSize::Medium,
            3 => WorldSize::Large,
            4 => WorldSize::Enormous,
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

/// Button bounds in screen coordinates
struct ButtonBounds {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
}

impl MainMenu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        // Load wallpaper texture for background and get its dimensions
        let (wallpaper_texture, wallpaper_width, wallpaper_height) = load_texture_from_bytes_with_size(device, queue, WALLPAPER, "menu_wallpaper");
        let wallpaper_view = wallpaper_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let button_texture = load_texture_from_bytes(device, queue, BUTTON_TEXTURE, "menu_button");
        let button_view = button_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sampler for wallpaper - ClampToEdge to prevent repeating
        let wallpaper_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,  // Use linear for smoother wallpaper
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Sampler for buttons - Nearest for pixelated look
        let button_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
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

        let wallpaper_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&wallpaper_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&wallpaper_sampler),
                },
            ],
            label: Some("menu_wallpaper_bind_group"),
        });

        let button_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&button_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&button_sampler),
                },
            ],
            label: Some("menu_button_bind_group"),
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
        viewport.update(queue, glyphon::Resolution { width, height });

        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

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
            "Tiny (8x8)",
            "Small (16x16)",
            "Medium (24x24)",
            "Large (32x32)",
            "Enormous (64x64)",
        ];
        for option in options {
            let mut buffer = Buffer::new(&mut font_system, Metrics::new(28.0, 38.0));
            buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
            buffer.set_text(&mut font_system, option, font_attrs, Shaping::Advanced);
            buffers.push(buffer);
        }

        // Load World button (button 5)
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(28.0, 38.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "Load World", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        // Quit button (button 6)
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(28.0, 38.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "Quit", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        // Instructions
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(25.0, 35.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "Click to select a world size", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        // World name input label
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(22.0, 32.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "World Name:", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        // World name input text (initially empty)
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(22.0, 32.0));
        buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        buffer.set_text(&mut font_system, "", font_attrs, Shaping::Advanced);
        buffers.push(buffer);

        Self {
            pipeline,
            wallpaper_bind_group,
            button_bind_group,
            vertex_buffer,
            index_buffer,
            hovered_option: None,
            window_width: width,
            window_height: height,
            wallpaper_width,
            wallpaper_height,
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            buffers,
            world_name_input: String::new(),
            world_name_error: None,
            input_focused: false,
        }
    }

    /// Get button bounds scaled to current window size
    fn get_button_bounds(&self, option_index: usize) -> ButtonBounds {
        // Scale positions based on window size
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        let base_y = 300.0 + option_index as f32 * 60.0;
        ButtonBounds {
            left: 380.0 * scale_x,
            right: 900.0 * scale_x,
            top: base_y * scale_y,
            bottom: (base_y + 45.0) * scale_y,
        }
    }

    /// Check if mouse position is over a button with proper coordinate handling
    pub fn check_hover(&mut self, mouse_x: f64, mouse_y: f64) -> bool {
        let mx = mouse_x as f32;
        let my = mouse_y as f32;

        for i in 0..7 {  // Changed from 5 to 7
            let bounds = self.get_button_bounds(i);
            if mx >= bounds.left && mx <= bounds.right && my >= bounds.top && my <= bounds.bottom {
                self.hovered_option = Some(i);
                return true;
            }
        }

        self.hovered_option = None;
        false
    }

    /// Handle mouse click with proper coordinate handling
    /// Returns the selected world size if a button was clicked and world name is valid
    pub fn handle_click(&mut self, mouse_x: f64, mouse_y: f64) -> Option<WorldSize> {
        let mx = mouse_x as f32;
        let my = mouse_y as f32;

        for i in 0..5 {
            let bounds = self.get_button_bounds(i);
            if mx >= bounds.left && mx <= bounds.right && my >= bounds.top && my <= bounds.bottom {
                // Validate world name before allowing creation
                if self.validate_world_name() {
                    return Some(WorldSize::from_index(i));
                } else {
                    return None;  // Validation failed
                }
            }
        }

        None
    }

    /// Get which button (by index) was clicked, if any
    /// Returns button index 0-6, or None if no button clicked
    pub fn get_clicked_button(&self, mouse_x: f64, mouse_y: f64) -> Option<usize> {
        let mx = mouse_x as f32;
        let my = mouse_y as f32;

        // Check all 7 buttons (5 world sizes + Load World + Quit)
        for i in 0..7 {
            let bounds = self.get_button_bounds(i);
            if mx >= bounds.left && mx <= bounds.right && my >= bounds.top && my <= bounds.bottom {
                return Some(i);
            }
        }

        None
    }

    /// Update window dimensions on resize
    pub fn resize(&mut self, queue: &wgpu::Queue, new_width: u32, new_height: u32) {
        self.window_width = new_width;
        self.window_height = new_height;
        self.viewport.update(queue, glyphon::Resolution {
            width: new_width,
            height: new_height
        });

        // Update text buffer sizes
        for buffer in &mut self.buffers {
            buffer.set_size(&mut self.font_system, Some(new_width as f32), Some(new_height as f32));
        }
    }

    /// Check if text input was clicked and update focus state
    pub fn check_input_click(&mut self, x: f64, y: f64) -> bool {
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        let input_x = 500.0 * scale_x;
        let input_width = 400.0 * scale_x;
        let input_y = 635.0 * scale_y;
        let input_height = 30.0 * scale_y;

        let clicked = x as f32 >= input_x && x as f32 <= input_x + input_width
            && y as f32 >= input_y && y as f32 <= input_y + input_height;

        let old_focused = self.input_focused;
        self.input_focused = clicked;

        old_focused != self.input_focused
    }

    /// Handle text input
    pub fn handle_text_input(&mut self, character: char) {
        if self.input_focused && self.world_name_input.len() < 30 && !character.is_control() {
            self.world_name_error = None;
            self.world_name_input.push(character);

            let font_attrs = Attrs::new().family(Family::Name("Minecraft"));
            self.buffers[11].set_text(&mut self.font_system, &self.world_name_input, font_attrs, Shaping::Advanced);
        }
    }

    /// Handle backspace
    pub fn handle_backspace(&mut self) {
        if self.input_focused && !self.world_name_input.is_empty() {
            self.world_name_error = None;
            self.world_name_input.pop();

            let font_attrs = Attrs::new().family(Family::Name("Minecraft"));
            self.buffers[11].set_text(&mut self.font_system, &self.world_name_input, font_attrs, Shaping::Advanced);
        }
    }

    /// Validate world name and return true if valid
    fn validate_world_name(&mut self) -> bool {
        self.world_name_error = None;

        if self.world_name_input.trim().is_empty() {
            self.world_name_error = Some("World name cannot be empty".to_string());
            return false;
        }

        if let Ok(worlds) = crate::save::list_worlds() {
            if worlds.iter().any(|w| w == self.world_name_input.trim()) {
                self.world_name_error = Some("World name already exists".to_string());
                return false;
            }
        }

        true
    }

    /// Get the world name
    pub fn get_world_name(&self) -> String {
        self.world_name_input.trim().to_string()
    }

    /// Check if input is focused
    pub fn is_input_focused(&self) -> bool {
        self.input_focused
    }

    /// Calculate texture coordinates for crop-to-fit behavior (like CSS object-fit: cover)
    fn calculate_crop_coords(&self) -> (f32, f32, f32, f32) {
        let window_aspect = self.window_width as f32 / self.window_height as f32;
        let image_aspect = self.wallpaper_width as f32 / self.wallpaper_height as f32;

        if window_aspect > image_aspect {
            // Window is wider - crop top and bottom of image
            let v_range = image_aspect / window_aspect;
            let v_min = (1.0 - v_range) / 2.0;
            let v_max = (1.0 + v_range) / 2.0;
            (0.0, v_min, 1.0, v_max)
        } else {
            // Window is taller - crop left and right of image
            let u_range = window_aspect / image_aspect;
            let u_min = (1.0 - u_range) / 2.0;
            let u_max = (1.0 + u_range) / 2.0;
            (u_min, 0.0, u_max, 1.0)
        }
    }

    /// Generate background geometry - returns number of indices
    pub fn update_geometry(&mut self, queue: &wgpu::Queue) -> usize {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Calculate crop-to-fit texture coordinates
        let (u_min, v_min, u_max, v_max) = self.calculate_crop_coords();

        // Background (single wallpaper image cropped to fit with dark overlay)
        // Use calculated texture coordinates to crop the image while maintaining aspect ratio
        // Use dark color multiplier [0.4, 0.4, 0.4, 1.0] to darken the wallpaper
        let bg_verts = [
            MenuVertex { position: [-1.0, -1.0], tex_coords: [u_min, v_max], color: [0.4, 0.4, 0.4, 1.0] },
            MenuVertex { position: [ 1.0, -1.0], tex_coords: [u_max, v_max], color: [0.4, 0.4, 0.4, 1.0] },
            MenuVertex { position: [ 1.0,  1.0], tex_coords: [u_max, v_min], color: [0.4, 0.4, 0.4, 1.0] },
            MenuVertex { position: [-1.0,  1.0], tex_coords: [u_min, v_min], color: [0.4, 0.4, 0.4, 1.0] },
        ];
        vertices.extend_from_slice(&bg_verts);
        indices.extend_from_slice(&[0u32, 1, 2, 2, 3, 0]);

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        indices.len()
    }

    /// Update button geometry - returns (start_index, num_indices) for buttons
    pub fn update_button_geometry(&mut self, queue: &wgpu::Queue) -> (usize, usize) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let to_ndc_x = |x: f32| (x / self.window_width as f32) * 2.0 - 1.0;
        let to_ndc_y = |y: f32| -((y / self.window_height as f32) * 2.0 - 1.0);

        // Draw button backgrounds using button texture
        for i in 0..7 {
            let bounds = self.get_button_bounds(i);

            // Only highlight if mouse is hovering over this button
            let color = if Some(i) == self.hovered_option {
                [1.2, 1.2, 1.2, 1.0] // Brighter when hovering
            } else {
                [1.0, 1.0, 1.0, 1.0] // Normal
            };

            let left = to_ndc_x(bounds.left);
            let right = to_ndc_x(bounds.right);
            let top = to_ndc_y(bounds.top);
            let bottom = to_ndc_y(bounds.bottom);

            let base = vertices.len() as u32;
            // Map entire button texture to each button
            let button_verts = [
                MenuVertex { position: [left, bottom], tex_coords: [0.0, 1.0], color },
                MenuVertex { position: [right, bottom], tex_coords: [1.0, 1.0], color },
                MenuVertex { position: [right, top], tex_coords: [1.0, 0.0], color },
                MenuVertex { position: [left, top], tex_coords: [0.0, 0.0], color },
            ];
            vertices.extend_from_slice(&button_verts);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        // Add input box background
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;
        let input_x = 500.0 * scale_x;
        let input_width = 400.0 * scale_x;
        let input_y = 635.0 * scale_y;
        let input_height = 30.0 * scale_y;

        let input_color = if self.input_focused {
            [1.0, 1.0, 1.0, 1.0]  // White when focused
        } else {
            [0.8, 0.8, 0.8, 1.0]  // Slightly dimmed when not focused
        };

        let left = to_ndc_x(input_x);
        let right = to_ndc_x(input_x + input_width);
        let top = to_ndc_y(input_y);
        let bottom = to_ndc_y(input_y + input_height);

        let base = vertices.len() as u32;
        let input_verts = [
            MenuVertex { position: [left, bottom], tex_coords: [0.0, 1.0], color: input_color },
            MenuVertex { position: [right, bottom], tex_coords: [1.0, 1.0], color: input_color },
            MenuVertex { position: [right, top], tex_coords: [1.0, 0.0], color: input_color },
            MenuVertex { position: [left, top], tex_coords: [0.0, 0.0], color: input_color },
        ];
        vertices.extend_from_slice(&input_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

        // Write to buffer at offset after background (6 indices for background quad)
        let vertex_offset = 4 * std::mem::size_of::<MenuVertex>(); // 4 vertices for background
        let index_offset = 6 * std::mem::size_of::<u32>(); // 6 indices for background

        queue.write_buffer(&self.vertex_buffer, vertex_offset as u64, bytemuck::cast_slice(&vertices));

        // Adjust indices to account for the 4 background vertices already in the buffer
        let adjusted_indices: Vec<u32> = indices.iter().map(|&i| i + 4).collect();
        queue.write_buffer(&self.index_buffer, index_offset as u64, bytemuck::cast_slice(&adjusted_indices));

        (6, indices.len()) // Start at index 6, return number of button indices
    }

    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue, bg_indices: usize) {
        // Update button geometry
        let (button_start, button_count) = self.update_button_geometry(queue);

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

            // Draw background with wallpaper texture (cropped and darkened)
            render_pass.set_bind_group(0, &self.wallpaper_bind_group, &[]);
            render_pass.draw_indexed(0..bg_indices as u32, 0, 0..1);

            // Draw buttons with button texture
            render_pass.set_bind_group(0, &self.button_bind_group, &[]);
            render_pass.draw_indexed(button_start as u32..(button_start + button_count) as u32, 0, 0..1);
        }

        // Scale text positions based on window size
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        // Update instructions buffer with error if present (do this BEFORE creating text_areas)
        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));
        let instruction_color = if let Some(ref error) = self.world_name_error {
            self.buffers[9].set_text(&mut self.font_system, error, font_attrs, Shaping::Advanced);
            GlyphonColor::rgb(255, 80, 80)  // Red for error
        } else {
            self.buffers[9].set_text(&mut self.font_system, "Click to select a world size", font_attrs, Shaping::Advanced);
            GlyphonColor::rgb(153, 153, 153)  // Gray for normal
        };

        // Prepare text areas
        let mut text_areas = vec![
            TextArea {
                buffer: &self.buffers[0],
                left: 340.0 * scale_x,
                top: 100.0 * scale_y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: GlyphonColor::rgb(255, 255, 255),
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.buffers[1],
                left: 440.0 * scale_x,
                top: 200.0 * scale_y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: GlyphonColor::rgb(204, 204, 204),
                custom_glyphs: &[],
            },
        ];

        // Options - centered in buttons
        for i in 0..7 {
            let y = (300.0 + i as f32 * 60.0) * scale_y + 8.0 * scale_y; // Offset for vertical centering
            // Only highlight if mouse is hovering over this button
            let color = if Some(i) == self.hovered_option {
                GlyphonColor::rgb(255, 255, 0) // Yellow when hovering
            } else {
                GlyphonColor::rgb(230, 230, 230) // White for normal
            };

            text_areas.push(TextArea {
                buffer: &self.buffers[2 + i],
                left: 420.0 * scale_x,
                top: y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: color,
                custom_glyphs: &[],
            });
        }

        // Instructions (with error color if error present)
        text_areas.push(TextArea {
            buffer: &self.buffers[9],
            left: 335.0 * scale_x,
            top: 650.0 * scale_y,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
            default_color: instruction_color,
            custom_glyphs: &[],
        });

        // World name input label
        text_areas.push(TextArea {
            buffer: &self.buffers[10],
            left: 510.0 * scale_x,
            top: 610.0 * scale_y,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
            default_color: GlyphonColor::rgb(200, 200, 200),
            custom_glyphs: &[],
        });

        // World name input text
        let input_color = if self.input_focused {
            GlyphonColor::rgb(255, 255, 255)
        } else {
            GlyphonColor::rgb(180, 180, 180)
        };
        text_areas.push(TextArea {
            buffer: &self.buffers[11],
            left: 510.0 * scale_x,
            top: 638.0 * scale_y,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
            default_color: input_color,
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

/// Load texture and return the texture along with its dimensions
fn load_texture_from_bytes_with_size(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], label: &str) -> (wgpu::Texture, u32, u32) {
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

    (texture, dimensions.0, dimensions.1)
}

/// Pause menu actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseAction {
    Resume,
    ExitToMenu,
    ExitToDesktop,
}

/// Pause menu - rendered over the game world
pub struct PauseMenu {
    pipeline: wgpu::RenderPipeline,
    button_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    hovered_option: Option<usize>,
    window_width: u32,
    window_height: u32,

    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    buffers: Vec<Buffer>,
}

impl PauseMenu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let button_texture = load_texture_from_bytes(device, queue, BUTTON_TEXTURE, "pause_button");
        let button_view = button_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let button_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
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
            label: Some("pause_texture_bind_group_layout"),
        });

        let button_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&button_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&button_sampler),
                },
            ],
            label: Some("pause_button_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pause Menu Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("loading.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pause Menu Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pause Menu Pipeline"),
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
            label: Some("Pause Menu Vertex Buffer"),
            size: 8192,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pause Menu Index Buffer"),
            size: 8192,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Text rendering setup
        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(Vec::from(FONT));

        let swash_cache = SwashCache::new();
        let cache = glyphon::Cache::new(device);
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(queue, glyphon::Resolution { width, height });

        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));

        // Create text buffers for pause menu
        let mut buffers = Vec::new();

        // Title
        let mut title_buffer = Buffer::new(&mut font_system, Metrics::new(80.0, 100.0));
        title_buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        title_buffer.set_text(&mut font_system, "Game Paused", font_attrs, Shaping::Advanced);
        buffers.push(title_buffer);

        // Button labels
        let button_labels = ["Resume Game", "Exit to Main Menu", "Exit to Desktop"];
        for label in &button_labels {
            let mut buffer = Buffer::new(&mut font_system, Metrics::new(30.0, 40.0));
            buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
            buffer.set_text(&mut font_system, label, font_attrs, Shaping::Advanced);
            buffers.push(buffer);
        }

        let mut pause_menu = Self {
            pipeline,
            button_bind_group,
            vertex_buffer,
            index_buffer,
            hovered_option: None,
            window_width: width,
            window_height: height,
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            buffers,
        };

        // Create initial geometry
        pause_menu.update_geometry(queue);

        pause_menu
    }

    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.window_width = width;
        self.window_height = height;
        self.viewport.update(queue, glyphon::Resolution { width, height });

        // Update buffer sizes
        for buffer in &mut self.buffers {
            buffer.set_size(&mut self.font_system, Some(width as f32), Some(height as f32));
        }
    }

    fn get_button_bounds(&self, index: usize) -> ButtonBounds {
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        let y = (280.0 + index as f32 * 80.0) * scale_y;

        ButtonBounds {
            left: 390.0 * scale_x,
            right: 890.0 * scale_x,
            top: y,
            bottom: y + 50.0 * scale_y,
        }
    }

    pub fn check_hover(&mut self, x: f64, y: f64) -> bool {
        let old_hover = self.hovered_option;

        for i in 0..3 {
            let bounds = self.get_button_bounds(i);
            if x as f32 >= bounds.left && x as f32 <= bounds.right &&
                y as f32 >= bounds.top && y as f32 <= bounds.bottom {
                self.hovered_option = Some(i);
                return old_hover != self.hovered_option;
            }
        }

        self.hovered_option = None;
        old_hover != self.hovered_option
    }

    pub fn handle_click(&self, x: f64, y: f64) -> Option<PauseAction> {
        for i in 0..3 {
            let bounds = self.get_button_bounds(i);
            if x as f32 >= bounds.left && x as f32 <= bounds.right &&
                y as f32 >= bounds.top && y as f32 <= bounds.bottom {
                return Some(match i {
                    0 => PauseAction::Resume,
                    1 => PauseAction::ExitToMenu,
                    2 => PauseAction::ExitToDesktop,
                    _ => unreachable!(),
                });
            }
        }
        None
    }

    pub fn update_geometry(&mut self, queue: &wgpu::Queue) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Semi-transparent dark overlay - full screen quad
        let overlay_verts = [
            MenuVertex { position: [-1.0, -1.0], tex_coords: [0.0, 1.0], color: [0.0, 0.0, 0.0, 0.7] },
            MenuVertex { position: [ 1.0, -1.0], tex_coords: [1.0, 1.0], color: [0.0, 0.0, 0.0, 0.7] },
            MenuVertex { position: [ 1.0,  1.0], tex_coords: [1.0, 0.0], color: [0.0, 0.0, 0.0, 0.7] },
            MenuVertex { position: [-1.0,  1.0], tex_coords: [0.0, 0.0], color: [0.0, 0.0, 0.0, 0.7] },
        ];
        vertices.extend_from_slice(&overlay_verts);
        indices.extend_from_slice(&[0u32, 1, 2, 2, 3, 0]);

        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let to_ndc_x = |x: f32| (x / self.window_width as f32) * 2.0 - 1.0;
        let to_ndc_y = |y: f32| -((y / self.window_height as f32) * 2.0 - 1.0);

        // Buttons
        for i in 0..3 {
            let bounds = self.get_button_bounds(i);

            let brightness = if Some(i) == self.hovered_option { 1.2 } else { 1.0 };
            let color = [brightness, brightness, brightness, 1.0];

            let left = to_ndc_x(bounds.left);
            let right = to_ndc_x(bounds.right);
            let top = to_ndc_y(bounds.top);
            let bottom = to_ndc_y(bounds.bottom);

            let base = vertices.len() as u32;
            let button_verts = [
                MenuVertex { position: [left, bottom], tex_coords: [0.0, 1.0], color },
                MenuVertex { position: [right, bottom], tex_coords: [1.0, 1.0], color },
                MenuVertex { position: [right, top], tex_coords: [1.0, 0.0], color },
                MenuVertex { position: [left, top], tex_coords: [0.0, 0.0], color },
            ];
            vertices.extend_from_slice(&button_verts);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
    }

    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Don't call update_geometry here - it's called when hover changes

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pause Menu Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pause Menu Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Load existing frame
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.button_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..24, 0, 0..1);  // 4 quads * 6 indices
        }

        // Render text
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        let mut text_areas = vec![
            // Title
            TextArea {
                buffer: &self.buffers[0],
                left: 440.0 * scale_x,
                top: 120.0 * scale_y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: GlyphonColor::rgb(255, 255, 255),
                custom_glyphs: &[],
            },
        ];

        // Button labels
        for i in 0..3 {
            let y = (280.0 + i as f32 * 80.0) * scale_y + 8.0 * scale_y;  // +8 for vertical centering
            let color = if Some(i) == self.hovered_option {
                GlyphonColor::rgb(255, 255, 0)
            } else {
                GlyphonColor::rgb(230, 230, 230)
            };

            text_areas.push(TextArea {
                buffer: &self.buffers[1 + i],
                left: 480.0 * scale_x,
                top: y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: color,
                custom_glyphs: &[],
            });
        }

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

/// World selection menu for loading existing worlds
pub struct WorldSelectionMenu {
    pipeline: wgpu::RenderPipeline,
    wallpaper_bind_group: wgpu::BindGroup,
    button_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    hovered_option: Option<usize>,
    window_width: u32,
    window_height: u32,
    wallpaper_width: u32,
    wallpaper_height: u32,

    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    buffers: Vec<Buffer>,

    /// List of available worlds
    pub worlds: Vec<crate::save::WorldInfo>,
}

#[derive(Debug, Clone)]
pub enum WorldSelectionAction {
    LoadWorld(String),  // World name to load
    CreateNew,
    Back,
}

impl WorldSelectionMenu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        // Load wallpaper texture for background
        let (wallpaper_texture, wallpaper_width, wallpaper_height) = load_texture_from_bytes_with_size(device, queue, WALLPAPER, "world_select_wallpaper");
        let wallpaper_view = wallpaper_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let button_texture = load_texture_from_bytes(device, queue, BUTTON_TEXTURE, "world_select_button");
        let button_view = button_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let wallpaper_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let button_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
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
            label: Some("world_select_texture_bind_group_layout"),
        });

        let wallpaper_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&wallpaper_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&wallpaper_sampler),
                },
            ],
            label: Some("world_select_wallpaper_bind_group"),
        });

        let button_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&button_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&button_sampler),
                },
            ],
            label: Some("world_select_button_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WorldSelection Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("loading.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WorldSelection Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("WorldSelection Pipeline"),
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
            label: Some("WorldSelection Vertex Buffer"),
            size: (200 * std::mem::size_of::<MenuVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorldSelection Index Buffer"),
            size: (400 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut font_system = FontSystem::new();
        font_system.db_mut().load_font_data(FONT.to_vec());

        let swash_cache = SwashCache::new();
        let cache = glyphon::Cache::new(device);
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(queue, glyphon::Resolution { width, height });

        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        let font_attrs = Attrs::new().family(Family::Name("Minecraft"));

        // Load world list
        let worlds = match crate::save::list_worlds() {
            Ok(world_names) => {
                world_names.iter()
                    .filter_map(|name| {
                        match crate::save::get_world_info(name) {
                            Ok(info) => Some(info),
                            Err(e) => {
                                eprintln!("Failed to load world info for '{}': {}", name, e);
                                None
                            }
                        }
                    })
                    .collect()
            }
            Err(e) => {
                eprintln!("Failed to list worlds: {}", e);
                Vec::new()
            }
        };

        // Create text buffers
        let mut buffers = Vec::new();

        // Title
        let mut title_buffer = Buffer::new(&mut font_system, Metrics::new(50.0, 60.0));
        title_buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        title_buffer.set_text(&mut font_system, "Select World", font_attrs, Shaping::Advanced);
        buffers.push(title_buffer);

        // World buttons (up to 5 slots shown)
        for i in 0..5 {
            let mut buffer = Buffer::new(&mut font_system, Metrics::new(30.0, 40.0));
            buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));

            if i < worlds.len() {
                let world = &worlds[i];
                let text = format!("{}", world.name);
                buffer.set_text(&mut font_system, &text, font_attrs, Shaping::Advanced);
            } else {
                buffer.set_text(&mut font_system, "<Empty>", font_attrs, Shaping::Advanced);
            }
            buffers.push(buffer);
        }

        // "Create New World" button
        let mut create_buffer = Buffer::new(&mut font_system, Metrics::new(30.0, 40.0));
        create_buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        create_buffer.set_text(&mut font_system, "Create New World", font_attrs, Shaping::Advanced);
        buffers.push(create_buffer);

        // "Back" button
        let mut back_buffer = Buffer::new(&mut font_system, Metrics::new(30.0, 40.0));
        back_buffer.set_size(&mut font_system, Some(width as f32), Some(height as f32));
        back_buffer.set_text(&mut font_system, "Back", font_attrs, Shaping::Advanced);
        buffers.push(back_buffer);

        let mut menu = Self {
            pipeline,
            wallpaper_bind_group,
            button_bind_group,
            vertex_buffer,
            index_buffer,
            hovered_option: None,
            window_width: width,
            window_height: height,
            wallpaper_width,
            wallpaper_height,
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            buffers,
            worlds,
        };

        menu.update_geometry(queue);
        menu
    }

    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.window_width = width;
        self.window_height = height;
        self.viewport.update(queue, glyphon::Resolution { width, height });

        for buffer in &mut self.buffers {
            buffer.set_size(&mut self.font_system, Some(width as f32), Some(height as f32));
        }
    }

    fn get_button_bounds(&self, index: usize) -> ButtonBounds {
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        let y = (150.0 + index as f32 * 70.0) * scale_y;

        ButtonBounds {
            left: 340.0 * scale_x,
            right: 940.0 * scale_x,
            top: y,
            bottom: y + 50.0 * scale_y,
        }
    }

    pub fn check_hover(&mut self, x: f64, y: f64) -> bool {
        let old_hover = self.hovered_option;

        // Total buttons: world slots (5) + create new (1) + back (1) = 7
        for i in 0..7 {
            let bounds = self.get_button_bounds(i);
            if x as f32 >= bounds.left && x as f32 <= bounds.right &&
                y as f32 >= bounds.top && y as f32 <= bounds.bottom {
                self.hovered_option = Some(i);
                return old_hover != self.hovered_option;
            }
        }

        self.hovered_option = None;
        old_hover != self.hovered_option
    }

    pub fn handle_click(&self, x: f64, y: f64) -> Option<WorldSelectionAction> {
        for i in 0..7 {
            let bounds = self.get_button_bounds(i);
            if x as f32 >= bounds.left && x as f32 <= bounds.right &&
                y as f32 >= bounds.top && y as f32 <= bounds.bottom {
                // First 5 buttons are world slots
                if i < 5 {
                    if i < self.worlds.len() {
                        return Some(WorldSelectionAction::LoadWorld(self.worlds[i].name.clone()));
                    }
                    // Empty slot - do nothing
                    return None;
                }
                // Button 5 is "Create New World"
                else if i == 5 {
                    return Some(WorldSelectionAction::CreateNew);
                }
                // Button 6 is "Back"
                else if i == 6 {
                    return Some(WorldSelectionAction::Back);
                }
            }
        }
        None
    }

    pub fn update_geometry(&mut self, queue: &wgpu::Queue) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Calculate aspect ratio and dimensions for wallpaper
        let window_aspect = self.window_width as f32 / self.window_height as f32;
        let wallpaper_aspect = self.wallpaper_width as f32 / self.wallpaper_height as f32;

        let (u_min, u_max, v_min, v_max) = if window_aspect > wallpaper_aspect {
            let scale = wallpaper_aspect / window_aspect;
            let v_offset = (1.0 - scale) / 2.0;
            (0.0, 1.0, v_offset, v_offset + scale)
        } else {
            let scale = window_aspect / wallpaper_aspect;
            let u_offset = (1.0 - scale) / 2.0;
            (u_offset, u_offset + scale, 0.0, 1.0)
        };

        // Wallpaper background
        let bg_verts = [
            MenuVertex { position: [-1.0, -1.0], tex_coords: [u_min, v_max], color: [0.8, 0.8, 0.8, 1.0] },
            MenuVertex { position: [ 1.0, -1.0], tex_coords: [u_max, v_max], color: [0.8, 0.8, 0.8, 1.0] },
            MenuVertex { position: [ 1.0,  1.0], tex_coords: [u_max, v_min], color: [0.8, 0.8, 0.8, 1.0] },
            MenuVertex { position: [-1.0,  1.0], tex_coords: [u_min, v_min], color: [0.8, 0.8, 0.8, 1.0] },
        ];

        let base = vertices.len() as u32;
        vertices.extend_from_slice(&bg_verts);
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);

        let to_ndc_x = |x: f32| (x / self.window_width as f32) * 2.0 - 1.0;
        let to_ndc_y = |y: f32| -((y / self.window_height as f32) * 2.0 - 1.0);

        // Buttons (7 total)
        for i in 0..7 {
            let bounds = self.get_button_bounds(i);

            // Dim empty slots
            let brightness = if i < 5 && i >= self.worlds.len() {
                0.5  // Dimmed
            } else if Some(i) == self.hovered_option {
                1.2  // Highlighted
            } else {
                1.0  // Normal
            };

            let color = [brightness, brightness, brightness, 1.0];

            let left = to_ndc_x(bounds.left);
            let right = to_ndc_x(bounds.right);
            let top = to_ndc_y(bounds.top);
            let bottom = to_ndc_y(bounds.bottom);

            let base = vertices.len() as u32;
            let button_verts = [
                MenuVertex { position: [left, bottom], tex_coords: [0.0, 1.0], color },
                MenuVertex { position: [right, bottom], tex_coords: [1.0, 1.0], color },
                MenuVertex { position: [right, top], tex_coords: [1.0, 0.0], color },
                MenuVertex { position: [left, top], tex_coords: [0.0, 0.0], color },
            ];
            vertices.extend_from_slice(&button_verts);
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
    }

    pub fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WorldSelection Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("WorldSelection Render Pass"),
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

            // Draw wallpaper background
            render_pass.set_bind_group(0, &self.wallpaper_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);

            // Draw buttons
            render_pass.set_bind_group(0, &self.button_bind_group, &[]);
            render_pass.draw_indexed(6..48, 0, 0..1);  // 7 buttons * 6 indices = 42, starting at index 6
        }

        // Render text
        let scale_x = self.window_width as f32 / 1280.0;
        let scale_y = self.window_height as f32 / 720.0;

        let mut text_areas = vec![
            // Title
            TextArea {
                buffer: &self.buffers[0],
                left: 480.0 * scale_x,
                top: 60.0 * scale_y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: GlyphonColor::rgb(255, 255, 255),
                custom_glyphs: &[],
            },
        ];

        // World slot labels (5)
        for i in 0..5 {
            let y = (150.0 + i as f32 * 70.0) * scale_y + 8.0 * scale_y;

            let color = if i < self.worlds.len() {
                if Some(i) == self.hovered_option {
                    GlyphonColor::rgb(255, 255, 0)
                } else {
                    GlyphonColor::rgb(230, 230, 230)
                }
            } else {
                GlyphonColor::rgb(100, 100, 100)  // Dimmed for empty slots
            };

            text_areas.push(TextArea {
                buffer: &self.buffers[1 + i],
                left: 430.0 * scale_x,
                top: y,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
                default_color: color,
                custom_glyphs: &[],
            });
        }

        // "Create New World" button label
        let create_y = (150.0 + 5.0 * 70.0) * scale_y + 8.0 * scale_y;
        let create_color = if Some(5) == self.hovered_option {
            GlyphonColor::rgb(255, 255, 0)
        } else {
            GlyphonColor::rgb(230, 230, 230)
        };
        text_areas.push(TextArea {
            buffer: &self.buffers[6],
            left: 460.0 * scale_x,
            top: create_y,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
            default_color: create_color,
            custom_glyphs: &[],
        });

        // "Back" button label
        let back_y = (150.0 + 6.0 * 70.0) * scale_y + 8.0 * scale_y;
        let back_color = if Some(6) == self.hovered_option {
            GlyphonColor::rgb(255, 255, 0)
        } else {
            GlyphonColor::rgb(230, 230, 230)
        };
        text_areas.push(TextArea {
            buffer: &self.buffers[7],
            left: 580.0 * scale_x,
            top: back_y,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: self.window_width as i32, bottom: self.window_height as i32 },
            default_color: back_color,
            custom_glyphs: &[],
        });

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