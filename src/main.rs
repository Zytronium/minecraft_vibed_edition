mod block;
mod camera;
mod vertex;
mod world;
mod loading;
mod menu;
mod models;
mod ui;
mod save;
mod state;
mod app;

use winit::{
    event_loop::{EventLoop},
    application::ApplicationHandler,
};
use wgpu::util::DeviceExt;
use std::hash::{BuildHasher, Hasher};

/// Game mode - Creative or Spectator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    Creative,   // Gravity, collision, block interaction
    Spectator,  // No gravity, no collision, fly through blocks
}

/// Player physics state for Creative mode
pub struct PlayerState {
    velocity: glam::Vec3,
    on_ground: bool,
    is_flying: bool,
    physics_enabled: bool,  // Disable physics during world generation
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            velocity: glam::Vec3::ZERO,
            on_ground: false,
            is_flying: false,
            physics_enabled: false,  // Start disabled until we find ground
        }
    }
}

// Embed all textures at compile time
const GRASS_TOP: &[u8] = include_bytes!("../assets/textures/grass_top.png");
const GRASS_BOTTOM: &[u8] = include_bytes!("../assets/textures/grass_bottom.png");
const GRASS_SIDE: &[u8] = include_bytes!("../assets/textures/grass_side.png");
const DIRT: &[u8] = include_bytes!("../assets/textures/dirt.png");
const STONE: &[u8] = include_bytes!("../assets/textures/stone.png");
const COBBLESTONE: &[u8] = include_bytes!("../assets/textures/cobblestone.png");
const BEDROCK: &[u8] = include_bytes!("../assets/textures/bedrock.png");
const LOG_TOP_BOTTOM: &[u8] = include_bytes!("../assets/textures/log_top-bottom.png");
const LOG_SIDE: &[u8] = include_bytes!("../assets/textures/log_side.png");
const LEAVES: &[u8] = include_bytes!("../assets/textures/leaves.png");
const GLASS: &[u8] = include_bytes!("../assets/textures/glass.png");
const PLANKS: &[u8] = include_bytes!("../assets/textures/planks.png");
const NOT_FOUND: &[u8] = include_bytes!("../assets/textures/not_found.png");
const HOTBAR_SLOT: &[u8] = include_bytes!("../assets/textures/ui/hotbar_slot.png");
const HOTBAR_SLOT_SELECTED: &[u8] = include_bytes!("../assets/textures/ui/hotbar_slot_selected.png");

// Embed JSON data files at compile time
const BLOCKS_JSON: &str = include_str!("blocks.json");
const ITEMS_JSON: &str = include_str!("items.json");
const CRAFTING_JSON: &str = include_str!("crafting.json");

/// GPU buffers for a single chunk's mesh data
/// Storing meshes per-chunk avoids hitting GPU buffer size limits
pub struct ChunkMesh {
    opaque_vertex_buffer: wgpu::Buffer,
    opaque_index_buffer: wgpu::Buffer,
    opaque_num_indices: u32,
    transparent_vertex_buffer: wgpu::Buffer,
    transparent_index_buffer: wgpu::Buffer,
    transparent_num_indices: u32,
    chunk_x: i32,
    chunk_z: i32,
}

/// Main rendering state - holds all GPU resources and game state

/// Build a map of texture names to indices
pub fn build_texture_map(blocks: &[models::Block]) -> std::collections::HashMap<String, u32> {
    let mut texture_map = std::collections::HashMap::new();
    let mut next_index = 0u32;

    for block in blocks {
        // Collect all unique texture names from this block
        let textures = &block.textures;

        if let Some(ref all) = textures.all {
            texture_map.entry(all.clone()).or_insert_with(|| {
                let idx = next_index;
                next_index += 1;
                idx
            });
        }

        for texture_name in [
            &textures.top,
            &textures.bottom,
            &textures.north,
            &textures.south,
            &textures.east,
            &textures.west,
        ].iter().filter_map(|t| t.as_ref()) {
            texture_map.entry(texture_name.clone()).or_insert_with(|| {
                let idx = next_index;
                next_index += 1;
                idx
            });
        }
    }

    texture_map
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let mut app = app::App {
        state: None,
        menu: None,
        world_selection: None,  // NEW
        menu_ctx: None,
        pause_menu: None,
        game_state: app::GameState::Menu,
        last_render_time: std::time::Instant::now(),
        last_mouse_pos: None,
        last_autosave: std::time::Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}
