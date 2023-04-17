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
    zoom: f32,
    detail: u32
}

@group(0) @binding(0)
var<uniform> cam: Camera;

@group(0) @binding(1)
var t_gradient: texture_2d<f32>;
@group(0)@binding(2)
var s_gradient: sampler;

const PHI: f32 = 1.61803398874989484820459; // Φ = Golden Ratio 

fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(tan(distance(xy * PHI, xy) * seed) * xy.x);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let subsamples = 16u;

    var seed = 34.346;
    var avg_col = vec4(0.0, 0.0, 0.0, 1.0);

    for (var s: u32 = 0u; s < subsamples; s++) {
        let rand_x = mix(-0.5, 0.5, gold_noise(in.clip_position.xy, seed));
        seed += rand_x;

        let rand_y = mix(-0.5, 0.5, gold_noise(in.clip_position.xy, seed));
        seed += rand_y;

        let c: vec2<f32> = (in.clip_position.xy - cam.pos + vec2(rand_x, rand_y)) * cam.zoom;
        var z: vec2<f32> = vec2(0.0, 0.0);

        var breakpoint = 0u;

        for (var i: u32 = 0u; i < cam.detail; i++) {
            let z_squared: vec2<f32> = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
            z = c + z_squared;

            if (z.x * z.x + z.y * z.y > 4.0) {
                breakpoint = i;
                break;
            }
        }

        avg_col += textureSample(t_gradient, s_gradient, vec2((f32(breakpoint) - log2(max(1.0, log2(length(z))))) / f32(cam.detail), 0.0));
    }

    return vec4(avg_col.xyz / f32(subsamples), 1.0);
}