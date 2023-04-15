struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4(in.position, 1.0);
    out.pos = in.position;

    return out;
}

struct Camera {
    pos: vec2<f32>,
    mouse: vec2<f32>,
    zoom: f32,
    pad: vec2<i32>,
}

@group(0) @binding(0)
var<uniform> cam: Camera;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let max_step: u32 = 500u;
    let max_val: f32 = 2.0;

    let c: vec2<f32> = (in.clip_position.xy - cam.pos) * cam.zoom;
    var z: vec2<f32> = vec2(0.0, 0.0);
    var break_point: u32 = 0u;

    for (var i: u32 = 1u; i <= max_step; i++) {
        let z_squared: vec2<f32> = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
        z = c + z_squared;

        if (length(z) > max_val) {
            break_point = i;
            break;
        }
    }

    return vec4<f32>(vec3(mix(0.0, 1.0, f32(break_point) / f32(max_step)) * f32(break_point != 0u)), 1.0);
}