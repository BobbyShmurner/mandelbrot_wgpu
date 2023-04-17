struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) text_coord: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4(in.position, 1.0);
    out.text_coord = smoothstep(vec3(-1.0), vec3(1.0), in.position);

    return out;
}

@group(0) @binding(0)
var t_main: texture_2d<f32>;
@group(0) @binding(1)
var s_main: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_main, s_main, in.text_coord.xy);
}