use glam::{Mat4, Vec3};
use crate::world::CHUNK_WIDTH;

/// First-person camera for navigating the world
pub struct Camera {
    pub position: Vec3,     // Camera position in world space
    pub yaw: f32,           // Horizontal rotation (left/right)
    pub pitch: f32,         // Vertical rotation (up/down)
    pub aspect: f32,        // Aspect ratio (width/height)
    pub fov: f32,           // Field of view in radians
    pub near: f32,          // Near clipping plane
    pub far: f32,           // Far clipping plane
}

impl Camera {
    pub fn new(aspect: f32, world_size_chunks: i32) -> Self {
        Self {
            // Start camera in the middle of the world, elevated above terrain
            position: Vec3::new(
                (world_size_chunks * CHUNK_WIDTH as i32 / 2) as f32,
                80.0,  // Above the surface (surface is around Y 64)
                (world_size_chunks * CHUNK_WIDTH as i32 / 2) as f32
            ),
            yaw: -90.0_f32.to_radians(),  // Face forward (negative Z)
            pitch: 0.0,
            aspect,
            fov: 70.0_f32.to_radians(),
            near: 0.1,
            far: 500.0,  // Increased to see far chunks
        }
    }

    /// Calculate the combined view-projection matrix for the shader
    pub fn get_view_projection(&self) -> [[f32; 4]; 4] {
        // Calculate camera direction from yaw and pitch
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let front = Vec3::new(
            cos_pitch * cos_yaw,
            sin_pitch,
            cos_pitch * sin_yaw,
        ).normalize();

        // Create view matrix (camera transform)
        let view = Mat4::look_at_rh(
            self.position,
            self.position + front,
            Vec3::Y,
        );

        // Create projection matrix (perspective)
        let proj = Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far);

        // Combine and return as 2D array for shader
        (proj * view).to_cols_array_2d()
    }

    /// Get forward direction vector (ignoring pitch, for WASD movement)
    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, sin_yaw).normalize()
    }

    /// Get right direction vector (for A/D strafing)
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }
}
