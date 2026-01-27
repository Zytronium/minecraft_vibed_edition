// UI shader for 2D overlay rendering (crosshair, hotbar, etc.)

@group(0) @binding(0)
var textures: binding_array<texture_2d<f32>, 20>;

@group(0) @binding(1)
var texture_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) texture_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) texture_index: u32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Convert from normalized coordinates (0-1) to clip space (-1 to 1)
    // Y is flipped because screen space has Y=0 at top, clip space has Y=-1 at top
    out.clip_position = vec4<f32>(
        input.position.x * 2.0 - 1.0,
        1.0 - input.position.y * 2.0,
        0.0,
        1.0
    );

    out.tex_coords = input.tex_coords;
    out.color = input.color;
    out.texture_index = input.texture_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.texture_index == 0u) {
        // No texture - check if we should invert
        if (in.color.a < 0.0) {
            // Negative alpha means invert colors (crosshair)
            // Return white - the blend mode will handle inversion
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        } else {
            // Normal solid color
        return in.color;
        }
    } else {
        // Sample texture and multiply by color
        let tex_color = textureSample(textures[in.texture_index], texture_sampler, in.tex_coords);

        // Return texture color as-is (including transparency)
        return tex_color * in.color;
    }
}
