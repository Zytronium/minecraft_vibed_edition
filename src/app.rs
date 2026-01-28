// App module - contains the App struct and event handling
use std::sync::Arc;
use winit::{
    event::*,
    event_loop::ActiveEventLoop,
    window::Window,
    keyboard::{KeyCode, PhysicalKey},
    application::ApplicationHandler,
};
use wgpu;

use crate::state::State;
use crate::menu::{MainMenu, PauseMenu, PauseAction, WorldSelectionMenu, WorldSelectionAction, WorldCreationMenu};
use crate::save;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    Menu,
    WorldSelection,
    WorldCreation,  // NEW: For world creation screen
    Playing,
    Paused,
}

pub struct MenuContext {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub window: Arc<Window>,
    pub surface_format: wgpu::TextureFormat,
}


pub struct App {
    pub state: Option<State>,
    pub menu: Option<MainMenu>,
    pub world_selection: Option<WorldSelectionMenu>,
    pub world_creation: Option<WorldCreationMenu>,  // NEW
    pub menu_ctx: Option<MenuContext>,
    pub pause_menu: Option<PauseMenu>,
    pub game_state: GameState,
    pub last_render_time: std::time::Instant,
    pub last_mouse_pos: Option<(f64, f64)>,
    pub last_autosave: std::time::Instant,
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
                            if let Some((x, y)) = self.last_mouse_pos {
                                let button_index = menu.get_clicked_button(x, y);

                                match button_index {
                                    Some(0) => {
                                        // Load World
                                        println!("Load World button clicked");
                                        if let Some(ctx) = &self.menu_ctx {
                                            let world_selection = WorldSelectionMenu::new(
                                                &ctx.device,
                                                &ctx.queue,
                                                ctx.surface_format,
                                                ctx.window.inner_size().width,
                                                ctx.window.inner_size().height,
                                            );
                                            self.world_selection = Some(world_selection);
                                            self.game_state = GameState::WorldSelection;
                                        }
                                    }
                                    Some(1) => {
                                        // Create New World
                                        println!("Create New World button clicked");
                                        if let Some(menu) = self.menu.take() {
                                            if let Some(ctx) = &self.menu_ctx {
                                                let world_creation = WorldCreationMenu::new(
                                                    &ctx.device,
                                                    &ctx.queue,
                                                    ctx.surface_format,
                                                    ctx.window.inner_size().width,
                                                    ctx.window.inner_size().height,
                                                    menu.pipeline,
                                                    menu.wallpaper_bind_group,
                                                    menu.button_bind_group,
                                                    menu.wallpaper_width,
                                                    menu.wallpaper_height,
                                                );
                                                self.world_creation = Some(world_creation);
                                                self.game_state = GameState::WorldCreation;
                                            }
                                        }
                                    }
                                    Some(2) => {
                                        // Quit
                                        println!("Quit button clicked");
                                        event_loop.exit();
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            state: key_state,
                            ..
                        },
                        ..
                    } => {
                        if key_state == ElementState::Pressed {
                            event_loop.exit();
                        }
                    }
                    _ => {}
                }
            }
            GameState::WorldSelection => {
                if let Some(world_selection) = &mut self.world_selection {
                    if let Some(ctx) = &self.menu_ctx {
                        match event {
                            WindowEvent::CloseRequested => event_loop.exit(),
                            WindowEvent::Resized(physical_size) => {
                                world_selection.resize(&ctx.queue, physical_size.width, physical_size.height);

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
                            WindowEvent::RedrawRequested => {
                                if let Ok(output) = ctx.surface.get_current_texture() {
                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                    world_selection.render(&view, &ctx.device, &ctx.queue);
                                    output.present();
                                }
                            }
                            WindowEvent::CursorMoved { position, .. } => {
                                self.last_mouse_pos = Some((position.x, position.y));
                                if world_selection.check_hover(position.x, position.y) {
                                    world_selection.update_geometry(&ctx.queue);
                                    ctx.window.request_redraw();
                                }
                            }
                            WindowEvent::MouseInput {
                                state: ElementState::Pressed,
                                button: MouseButton::Left,
                                ..
                            } => {
                                if let Some(pos) = self.last_mouse_pos {
                                    if let Some(action) = world_selection.handle_click(pos.0, pos.1) {
                                        match action {
                                            WorldSelectionAction::LoadWorld(world_name) => {
                                                println!("Loading world: {}", world_name);

                                                match save::load_world(&world_name) {
                                                    Ok(world_save) => {
                                                        // Load the world
                                                        if let Some(ctx) = self.menu_ctx.take() {
                                                            self.state = Some(pollster::block_on(State::from_world_save(
                                                                ctx.window,
                                                                world_save,
                                                                ctx.instance,
                                                                ctx.surface
                                                            )));
                                                            self.world_selection = None;
                                                            self.menu = None;
                                                            self.game_state = GameState::Playing;
                                                            self.last_render_time = std::time::Instant::now();

                                                            // Lock cursor
                                                            if let Some(state) = &self.state {
                                                                state.window.set_cursor_visible(false);
                                                                let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                                                    .or_else(|_| state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
                                                            }

                                                            println!("World loaded successfully!");
                                                        }
                                                    }
                                                    Err(e) => {
                                                        eprintln!("Failed to load world: {}", e);
                                                    }
                                                }
                                            }
                                            WorldSelectionAction::CreateNew => {
                                                // Recreate main menu before returning
                                                let mut menu = MainMenu::new(
                                                    &ctx.device,
                                                    &ctx.queue,
                                                    ctx.surface_format,
                                                    ctx.window.inner_size().width,
                                                    ctx.window.inner_size().height
                                                );
                                                let bg_indices = menu.update_geometry(&ctx.queue);

                                                // Render initial menu
                                                if let Ok(output) = ctx.surface.get_current_texture() {
                                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                                    menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                                    output.present();
                                                }

                                                self.menu = Some(menu);
                                                self.world_selection = None;
                                                self.game_state = GameState::Menu;
                                                println!("Returning to main menu for new world creation");
                                            }
                                            WorldSelectionAction::Back => {
                                                // Recreate main menu before returning
                                                let mut menu = MainMenu::new(
                                                    &ctx.device,
                                                    &ctx.queue,
                                                    ctx.surface_format,
                                                    ctx.window.inner_size().width,
                                                    ctx.window.inner_size().height
                                                );
                                                let bg_indices = menu.update_geometry(&ctx.queue);

                                                // Render initial menu
                                                if let Ok(output) = ctx.surface.get_current_texture() {
                                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                                    menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                                    output.present();
                                                }

                                                self.menu = Some(menu);
                                                self.world_selection = None;
                                                self.game_state = GameState::Menu;
                                                println!("Returning to main menu");
                                            }
                                        }
                                    }
                                }
                            }
                            WindowEvent::KeyboardInput {
                                event: KeyEvent {
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    state: key_state,
                                    ..
                                },
                                ..
                            } => {
                                if key_state == ElementState::Pressed {
                                    // Recreate main menu before returning
                                    let mut menu = MainMenu::new(
                                        &ctx.device,
                                        &ctx.queue,
                                        ctx.surface_format,
                                        ctx.window.inner_size().width,
                                        ctx.window.inner_size().height
                                    );
                                    let bg_indices = menu.update_geometry(&ctx.queue);

                                    // Render initial menu
                                    if let Ok(output) = ctx.surface.get_current_texture() {
                                        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                        menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                        output.present();
                                    }

                                    self.menu = Some(menu);
                                    self.world_selection = None;
                                    self.game_state = GameState::Menu;
                                    println!("Returning to main menu");
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            GameState::WorldCreation => {
                if let Some(world_creation) = &mut self.world_creation {
                    if let Some(ctx) = &self.menu_ctx {
                        match event {
                            WindowEvent::CloseRequested => event_loop.exit(),
                            WindowEvent::Resized(physical_size) => {
                                world_creation.resize(&ctx.queue, physical_size.width, physical_size.height);

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
                            WindowEvent::RedrawRequested => {
                                if let Ok(output) = ctx.surface.get_current_texture() {
                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                    let bg_indices = world_creation.update_geometry(&ctx.queue);
                                    world_creation.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                    output.present();
                                }
                            }
                            WindowEvent::CursorMoved { position, .. } => {
                                self.last_mouse_pos = Some((position.x, position.y));
                                if world_creation.check_hover(position.x, position.y) {
                                    ctx.window.request_redraw();
                                }
                            }
                            WindowEvent::MouseInput {
                                state: ElementState::Pressed,
                                button: MouseButton::Left,
                                ..
                            } => {
                                if let Some(pos) = self.last_mouse_pos {
                                    // Check if clicking on input field
                                    if world_creation.check_input_click(pos.0, pos.1) {
                                        ctx.window.request_redraw();
                                    } else {
                                        let button_index = world_creation.get_clicked_button(pos.0, pos.1);

                                        match button_index {
                                            Some(0) | Some(1) | Some(2) | Some(3) | Some(4) => {
                                                // World size buttons
                                                if let Some(selected_size) = world_creation.handle_click(pos.0, pos.1) {
                                                    let world_name = world_creation.get_world_name();
                                                    println!("Starting world generation: {:?} chunks, name: {}", selected_size.get_chunks(), world_name);

                                                    if let Some(ctx) = self.menu_ctx.take() {
                                                        self.state = Some(pollster::block_on(State::new(
                                                            ctx.window,
                                                            selected_size,
                                                            ctx.instance,
                                                            ctx.surface,
                                                            world_name
                                                        )));
                                                        self.world_creation = None;
                                                        self.menu = None;
                                                        self.game_state = GameState::Playing;
                                                    }
                                                } else {
                                                    // Validation failed
                                                    ctx.window.request_redraw();
                                                }
                                            }
                                            Some(5) => {
                                                // Back button - recreate main menu
                                                let mut menu = MainMenu::new(
                                                    &ctx.device,
                                                    &ctx.queue,
                                                    ctx.surface_format,
                                                    ctx.window.inner_size().width,
                                                    ctx.window.inner_size().height
                                                );
                                                let bg_indices = menu.update_geometry(&ctx.queue);

                                                // Render initial menu
                                                if let Ok(output) = ctx.surface.get_current_texture() {
                                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                                    menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                                    output.present();
                                                }

                                                self.menu = Some(menu);
                                                self.world_creation = None;
                                                self.game_state = GameState::Menu;
                                                println!("Returning to main menu");
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            WindowEvent::KeyboardInput {
                                event: KeyEvent {
                                    text,
                                    physical_key: PhysicalKey::Code(key),
                                    state: key_state,
                                    ..
                                },
                                ..
                            } => {
                                if key_state == ElementState::Pressed {
                                    if world_creation.is_input_focused() {
                                        match key {
                                            KeyCode::Backspace => {
                                                world_creation.handle_backspace();
                                                ctx.window.request_redraw();
                                            }
                                            KeyCode::Escape => {
                                                // Recreate main menu before returning
                                                let mut menu = MainMenu::new(
                                                    &ctx.device,
                                                    &ctx.queue,
                                                    ctx.surface_format,
                                                    ctx.window.inner_size().width,
                                                    ctx.window.inner_size().height
                                                );
                                                let bg_indices = menu.update_geometry(&ctx.queue);

                                                // Render initial menu
                                                if let Ok(output) = ctx.surface.get_current_texture() {
                                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                                    menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                                    output.present();
                                                }

                                                self.menu = Some(menu);
                                                self.world_creation = None;
                                                self.game_state = GameState::Menu;
                                            }
                                            _ => {
                                                if let Some(text) = text {
                                                    for c in text.chars() {
                                                        world_creation.handle_text_input(c);
                                                    }
                                                    ctx.window.request_redraw();
                                                }
                                            }
                                        }
                                    } else if key == KeyCode::Escape {
                                        // Recreate main menu before returning
                                        let mut menu = MainMenu::new(
                                            &ctx.device,
                                            &ctx.queue,
                                            ctx.surface_format,
                                            ctx.window.inner_size().width,
                                            ctx.window.inner_size().height
                                        );
                                        let bg_indices = menu.update_geometry(&ctx.queue);

                                        // Render initial menu
                                        if let Ok(output) = ctx.surface.get_current_texture() {
                                            let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                            menu.render(&view, &ctx.device, &ctx.queue, bg_indices);
                                            output.present();
                                        }

                                        self.menu = Some(menu);
                                        self.world_creation = None;
                                        self.game_state = GameState::Menu;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            GameState::Playing => {
                // Check for close requested - handle outside the state borrow
                if matches!(event, WindowEvent::CloseRequested) {
                    self.state = None;
                    self.pause_menu = None;
                    event_loop.exit();
                    return;
                }

                // Track if we need to exit due to OOM (checked after borrow ends)
                let mut should_exit_oom = false;

                if let Some(state) = &mut self.state {
                    // Check for ESC key to pause BEFORE passing to state.input()
                    if let WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            state: key_state,
                            ..
                        },
                        ..
                    } = event {
                        if key_state == ElementState::Pressed {
                            // Transition to paused
                            self.game_state = GameState::Paused;

                            // Show cursor and release grab
                            state.window.set_cursor_visible(true);
                            let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::None);

                            // Create pause menu if it doesn't exist
                            if self.pause_menu.is_none() {
                                self.pause_menu = Some(PauseMenu::new(
                                    &state.device,
                                    &state.queue,
                                    state.config.format,
                                    state.size.width,
                                    state.size.height,
                                ));
                            }

                            println!("Game paused");
                            state.window.request_redraw();
                            return;  // Don't process this event further
                        }
                    }

                    if !state.input(&event) {
                        match event {
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
                                    Err(wgpu::SurfaceError::OutOfMemory) => {
                                        // Will handle after borrow ends
                                        should_exit_oom = true;
                                    }
                                    Err(e) => eprintln!("{:?}", e),
                                }

                                if !should_exit_oom {
                                    state.window.request_redraw();
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // Borrow has ended naturally here

                if should_exit_oom {
                    self.state = None;
                    self.pause_menu = None;
                    event_loop.exit();
                }
            }
            GameState::Paused => {
                // Check for close requested - handle outside the state borrow
                if matches!(event, WindowEvent::CloseRequested) {
                    self.state = None;
                    self.pause_menu = None;
                    event_loop.exit();
                    return;
                }

                // Track if we need to exit due to OOM (checked after borrow ends)
                let mut should_exit_oom = false;

                if let Some(state) = &mut self.state {
                    match event {
                        WindowEvent::Resized(physical_size) => {
                            state.resize(physical_size);
                            if let Some(pause_menu) = &mut self.pause_menu {
                                pause_menu.resize(&state.queue, physical_size.width, physical_size.height);
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            // Don't call state.render() - it presents the frame
                            // Instead, render both game and pause menu in one pass
                            match state.surface.get_current_texture() {
                                Ok(output) => {
                                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

                                    // Render the frozen game world
                                    let depth_texture = state.device.create_texture(&wgpu::TextureDescriptor {
                                        label: Some("Pause Depth Texture"),
                                        size: wgpu::Extent3d {
                                            width: state.config.width,
                                            height: state.config.height,
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

                                    let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                        label: Some("Pause Render Encoder"),
                                    });

                                    // Render 3D world (frozen)
                                    {
                                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: Some("Frozen World Render Pass"),
                                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                                        r: 0.5, g: 0.7, b: 1.0, a: 1.0,
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

                                        render_pass.set_bind_group(0, &state.camera_bind_group, &[]);
                                        render_pass.set_bind_group(1, &state.texture_bind_group, &[]);

                                        // Render opaque geometry
                                        render_pass.set_pipeline(&state.render_pipeline_opaque);
                                        for chunk_mesh in &state.chunk_meshes {
                                            if chunk_mesh.opaque_num_indices > 0 {
                                                render_pass.set_vertex_buffer(0, chunk_mesh.opaque_vertex_buffer.slice(..));
                                                render_pass.set_index_buffer(chunk_mesh.opaque_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                                render_pass.draw_indexed(0..chunk_mesh.opaque_num_indices, 0, 0..1);
                                            }
                                        }

                                        // Render transparent geometry
                                        render_pass.set_pipeline(&state.render_pipeline_transparent);
                                        for chunk_mesh in &state.chunk_meshes {
                                            if chunk_mesh.transparent_num_indices > 0 {
                                                render_pass.set_vertex_buffer(0, chunk_mesh.transparent_vertex_buffer.slice(..));
                                                render_pass.set_index_buffer(chunk_mesh.transparent_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                                                render_pass.draw_indexed(0..chunk_mesh.transparent_num_indices, 0, 0..1);
                                            }
                                        }
                                    }

                                    // Render UI (hotbar/crosshair) over the game
                                    {
                                        let mut ui_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: Some("Frozen UI Render Pass"),
                                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                                view: &view,
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

                                        state.ui_renderer.render(&mut ui_render_pass);
                                    }

                                    state.queue.submit(std::iter::once(encoder.finish()));

                                    // Now render pause menu over everything
                                    if let Some(pause_menu) = &mut self.pause_menu {
                                        pause_menu.render(&view, &state.device, &state.queue);
                                    }

                                    output.present();
                                }
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    // Will handle after borrow ends
                                    should_exit_oom = true;
                                }
                                Err(e) => eprintln!("{:?}", e),
                            }

                            if !should_exit_oom {
                                state.window.request_redraw();
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            self.last_mouse_pos = Some((position.x, position.y));
                            if let Some(pause_menu) = &mut self.pause_menu {
                                if pause_menu.check_hover(position.x, position.y) {
                                    pause_menu.update_geometry(&state.queue);
                                    state.window.request_redraw();
                                }
                            }
                        }
                        WindowEvent::MouseInput {
                            state: mouse_state,
                            button: MouseButton::Left,
                            ..
                        } => {
                            if mouse_state == ElementState::Pressed {
                                if let Some(pause_menu) = &self.pause_menu {
                                    if let Some((x, y)) = self.last_mouse_pos {
                                        if let Some(action) = pause_menu.handle_click(x, y) {
                                            match action {
                                                PauseAction::Resume => {
                                                    // Resume game
                                                    self.game_state = GameState::Playing;
                                                    self.last_render_time = std::time::Instant::now();  // Reset time to prevent huge dt
                                                    state.window.set_cursor_visible(false);
                                                    let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                                        .or_else(|_| state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
                                                    println!("Game resumed");
                                                }
                                                PauseAction::ExitToMenu => {
                                                    println!("Exiting to main menu");

                                                    // Save the world
                                                    if let Err(e) = state.save_world(true) {
                                                        eprintln!("Failed to save world: {}", e);
                                                    } else {
                                                        println!("World saved successfully");
                                                    }

                                                    // Get window reference before dropping state
                                                    let window = state.window.clone();
                                                    window.set_cursor_visible(true);
                                                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);

                                                    // Drop the game state to free resources
                                                    self.state = None;
                                                    self.pause_menu = None;

                                                    // Create new menu context using the same window
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

                                                    // Transition to menu state
                                                    self.menu = Some(menu);
                                                    self.menu_ctx = Some(MenuContext {
                                                        instance,
                                                        surface,
                                                        device,
                                                        queue,
                                                        window,
                                                        surface_format,
                                                    });
                                                    self.game_state = GameState::Menu;
                                                    println!("Transitioned to main menu");
                                                }
                                                PauseAction::ExitToDesktop => {
                                                    println!("Exiting to desktop");

                                                    // Save the world
                                                    if let Some(state) = &self.state {
                                                        if let Err(e) = state.save_world(true) {
                                                            eprintln!("Failed to save world: {}", e);
                                                        }
                                                    }

                                                    // Clean up and exit
                                                    self.state = None;
                                                    self.pause_menu = None;
                                                    event_loop.exit();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        WindowEvent::KeyboardInput {
                            event: KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                state: key_state,
                                ..
                            },
                            ..
                        } => {
                            if key_state == ElementState::Pressed {
                                // Resume game on ESC
                                self.game_state = GameState::Playing;
                                self.last_render_time = std::time::Instant::now();  // Reset time to prevent huge dt
                                state.window.set_cursor_visible(false);
                                let _ = state.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                    .or_else(|_| state.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
                                println!("Game resumed");
                            }
                        }
                        _ => {}
                    }
                }
                // Borrow has ended naturally here

                if should_exit_oom {
                    self.state = None;
                    self.pause_menu = None;
                    event_loop.exit();
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
            GameState::WorldSelection | GameState::WorldCreation => {
                if let Some(ctx) = &self.menu_ctx {
                    ctx.window.request_redraw();
                }
            }
            GameState::Playing | GameState::Paused => {
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
        }
    }
}
