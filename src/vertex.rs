/// Vertex data sent to the GPU for rendering
/// Each vertex represents a corner of a triangle in the mesh
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],      // 3D position in world space
    pub tex_coords: [f32; 2],    // UV coordinates for texture mapping (0.0 to 1.0)
    pub brightness: f32,         // Lighting value (darker for bottom/sides, brighter for top)
    pub texture_index: u32,      // Which texture to use from our texture array
}

impl Vertex {
    /// Describes the memory layout of vertex data for the GPU
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position attribute (location 0)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinate attribute (location 1)
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Brightness attribute (location 2)
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                // Texture index attribute (location 3)
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// Generate the 4 vertices for a single cube face
/// Returns vertices in counter-clockwise order for proper backface culling
pub fn get_face_vertices(pos: [f32; 3], face: usize, texture_index: u32) -> [Vertex; 4] {
    let [x, y, z] = pos;

    // Different brightness for different faces (Minecraft-style ambient occlusion)
    let brightness = match face {
        3 => 1.0,       // Top - brightest (full sunlight)
        4 | 5 => 0.8,   // Front/Back - medium-bright
        0 | 1 => 0.6,   // Left/Right - medium-dark
        2 => 0.5,       // Bottom - darkest
        _ => 1.0,
    };

    // UV coordinates map texture onto face
    // (0,0) = top-left, (1,1) = bottom-right in texture space
    match face {
        0 => [ // Left face (-X)
            Vertex { position: [x, y, z], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z], tex_coords: [0.0, 0.0], brightness, texture_index },
        ],
        1 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
        ],
        2 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y, z], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
        ],
        3 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z], tex_coords: [0.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [1.0, 1.0], brightness, texture_index },
        ],
        4 => [ // Back face (-Z)
            Vertex { position: [x, y, z], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y, z], tex_coords: [0.0, 1.0], brightness, texture_index },
        ],
        5 => [ // Front face (+Z)
            Vertex { position: [x + 1.0, y, z + 1.0], tex_coords: [1.0, 1.0], brightness, texture_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], tex_coords: [1.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y + 1.0, z + 1.0], tex_coords: [0.0, 0.0], brightness, texture_index },
            Vertex { position: [x, y, z + 1.0], tex_coords: [0.0, 1.0], brightness, texture_index },
        ],
        _ => unreachable!(),
    }
}
