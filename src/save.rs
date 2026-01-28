/// World save/load system
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::io::{Read, Write};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use crate::block::BlockType;
use crate::world::{CHUNK_WIDTH, CHUNK_HEIGHT};

/// Player state to save/load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPlayerState {
    // Camera/Position
    pub position: [f32; 3],  // Vec3 as array for serde
    pub yaw: f32,
    pub pitch: f32,

    // Physics state
    pub velocity: [f32; 3],
    pub on_ground: bool,
    pub is_flying: bool,
    pub physics_enabled: bool,

    // Game mode
    pub game_mode: SavedGameMode,

    // Hotbar
    pub selected_slot: usize,
    pub hotbar_blocks: [u16; 9],  // Block IDs
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SavedGameMode {
    Creative,
    Spectator,
}

/// A single chunk's data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedChunk {
    pub offset_x: i32,
    pub offset_z: i32,
    // Flattened block data for efficient serialization
    // blocks[x][y][z] -> blocks[x * CHUNK_HEIGHT * CHUNK_WIDTH + y * CHUNK_WIDTH + z]
    pub blocks: Vec<u16>,  // BlockType IDs
}

impl SavedChunk {
    /// Convert from game chunk to saved format
    pub fn from_chunk(offset_x: i32, offset_z: i32, blocks: &[[[BlockType; CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH]) -> Self {
        let mut flat_blocks = Vec::with_capacity(CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_WIDTH);

        for x in 0..CHUNK_WIDTH {
            for y in 0..CHUNK_HEIGHT {
                for z in 0..CHUNK_WIDTH {
                    flat_blocks.push(blocks[x][y][z].id);
                }
            }
        }

        Self {
            offset_x,
            offset_z,
            blocks: flat_blocks,
        }
    }

    /// Convert back to game chunk format
    pub fn to_blocks(&self) -> [[[BlockType; CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH] {
        let mut blocks = [[[BlockType::air(); CHUNK_WIDTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];

        let mut idx = 0;
        for x in 0..CHUNK_WIDTH {
            for y in 0..CHUNK_HEIGHT {
                for z in 0..CHUNK_WIDTH {
                    blocks[x][y][z] = BlockType::new(self.blocks[idx]);
                    idx += 1;
                }
            }
        }

        blocks
    }
}

/// Complete world save data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSave {
    // Metadata
    pub world_name: String,
    pub version: u32,  // Save format version
    pub created_at: u64,  // Unix timestamp
    pub last_played: u64,  // Unix timestamp
    pub play_time_seconds: u64,

    // World generation
    pub seed: u64,
    pub world_size_chunks: i32,

    // Player state
    pub player: SavedPlayerState,

    // World data
    pub chunks: Vec<SavedChunk>,
}

/// Get the base save directory for the current platform
pub fn get_save_directory() -> Result<PathBuf, std::io::Error> {
    let base_dir = if cfg!(target_os = "windows") {
        // Windows: %APPDATA%\.minecraft_vibed\saves
        let appdata = std::env::var("APPDATA")
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "APPDATA not found"))?;
        PathBuf::from(appdata).join(".minecraft_vibed").join("saves")
    } else if cfg!(target_os = "macos") {
        // macOS: ~/Library/Application Support/.minecraft_vibed/saves
        let home = std::env::var("HOME")
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "HOME not found"))?;
        PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join(".minecraft_vibed")
            .join("saves")
    } else {
        // Linux/Unix: ~/.minecraft_vibed/saves
        let home = std::env::var("HOME")
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "HOME not found"))?;
        PathBuf::from(home).join(".minecraft_vibed").join("saves")
    };

    // Create directory if it doesn't exist
    fs::create_dir_all(&base_dir)?;

    Ok(base_dir)
}

/// Get the directory for a specific world
pub fn get_world_directory(world_name: &str) -> Result<PathBuf, std::io::Error> {
    let save_dir = get_save_directory()?;
    let world_dir = save_dir.join(sanitize_filename(world_name));

    // Create directory if it doesn't exist
    fs::create_dir_all(&world_dir)?;

    Ok(world_dir)
}

/// Sanitize a world name to be a valid filename
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

/// Save a world to disk
pub fn save_world(world_save: &WorldSave, compress: bool) -> Result<(), std::io::Error> {
    let world_dir = get_world_directory(&world_save.world_name)?;
    let level_file = world_dir.join("level.dat");

    // Serialize to binary
    let serialized = bincode::serialize(world_save)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Write to file (with optional compression)
    if compress {
        let file = fs::File::create(&level_file)?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(&serialized)?;
        encoder.finish()?;
    } else {
        fs::write(&level_file, serialized)?;
    }

    println!("World '{}' saved to: {:?}", world_save.world_name, level_file);
    Ok(())
}

/// Load a world from disk
pub fn load_world(world_name: &str) -> Result<WorldSave, std::io::Error> {
    let world_dir = get_world_directory(world_name)?;
    let level_file = world_dir.join("level.dat");

    if !level_file.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("World '{}' not found", world_name)
        ));
    }

    // Read file
    let file_data = fs::read(&level_file)?;

    // Try to decompress (if it fails, assume it's uncompressed)
    let decompressed = match try_decompress(&file_data) {
        Ok(data) => data,
        Err(_) => file_data,  // Not compressed
    };

    // Deserialize
    let world_save: WorldSave = bincode::deserialize(&decompressed)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    println!("World '{}' loaded from: {:?}", world_name, level_file);
    Ok(world_save)
}

/// Try to decompress gzip data
fn try_decompress(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

/// List all saved worlds
pub fn list_worlds() -> Result<Vec<String>, std::io::Error> {
    let save_dir = get_save_directory()?;

    let mut worlds = Vec::new();

    if save_dir.exists() {
        for entry in fs::read_dir(save_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        // Check if level.dat exists
                        if path.join("level.dat").exists() {
                            worlds.push(name_str.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(worlds)
}

/// Delete a saved world
pub fn delete_world(world_name: &str) -> Result<(), std::io::Error> {
    let world_dir = get_world_directory(world_name)?;

    if world_dir.exists() {
        fs::remove_dir_all(&world_dir)?;
        println!("World '{}' deleted", world_name);
    }

    Ok(())
}

/// Get world metadata without loading the entire world
pub fn get_world_info(world_name: &str) -> Result<WorldInfo, std::io::Error> {
    let world_save = load_world(world_name)?;

    Ok(WorldInfo {
        name: world_save.world_name,
        created_at: world_save.created_at,
        last_played: world_save.last_played,
        play_time_seconds: world_save.play_time_seconds,
        world_size_chunks: world_save.world_size_chunks,
    })
}

/// Minimal world information for display
#[derive(Debug, Clone)]
pub struct WorldInfo {
    pub name: String,
    pub created_at: u64,
    pub last_played: u64,
    pub play_time_seconds: u64,
    pub world_size_chunks: i32,
}

impl WorldInfo {
    /// Format play time as HH:MM:SS
    pub fn format_play_time(&self) -> String {
        let hours = self.play_time_seconds / 3600;
        let minutes = (self.play_time_seconds % 3600) / 60;
        let seconds = self.play_time_seconds % 60;
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
}
