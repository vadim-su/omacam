struct Uniforms {
    width: u32,
    height: u32,
    bg_mode: u32,        // 0 = None, 1 = Blur, 2 = Image, 3 = Color
    frame_index: u32,    // 0 resets temporal history
    blur_strength: f32,  // 0.0..1.0
    studio_light: f32,   // 0.0..1.0
    bg_image_blur: f32,  // 0.0..1.0
    sharpness: f32,      // 0.0..1.0 (2-Iter Richardson-Lucy Optical Deconvolution + 3-Band CAS/Clarity)
    centroid: vec2<f32>, // normalized [0..1, 0..1] face focal point (in sensor coords)
    mask_size: vec2<f32>,// e.g. (768.0, 432.0) for ResNet50 Vulkan or (512.0, 288.0) for MobileNetV3
    bg_color: vec4<f32>, // rgb in 0..1
    crop: vec4<f32>,     // (cx, cy, zoom, has_fgr_decontam)
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var lin_sampler: sampler;
@group(0) @binding(2) var y_tex: texture_2d<f32>;
@group(0) @binding(3) var uv_tex: texture_2d<f32>;
@group(0) @binding(4) var src_rgba_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var src_rgba_in: texture_2d<f32>;
@group(0) @binding(6) var bg_img_tex: texture_2d<f32>;
@group(0) @binding(7) var alpha_tex: texture_2d<f32>;
@group(0) @binding(8) var half_bg_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(9) var half_bg_in: texture_2d<f32>;
@group(0) @binding(10) var bokeh_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(11) var bokeh_in: texture_2d<f32>;
@group(0) @binding(12) var<storage, read_write> nv12_out: array<u32>;
@group(0) @binding(13) var fgr_tex: texture_2d<f32>;
// Temporal micro-stabilizer + 2-Iteration Richardson-Lucy Optical Deconvolution textures
@group(0) @binding(14) var prev_clean_in: texture_2d<f32>;
@group(0) @binding(15) var clean_rgba_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(16) var clean_rgba_in: texture_2d<f32>;
@group(0) @binding(17) var feat0_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(18) var feat1_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var feat0_in: texture_2d<f32>;
@group(0) @binding(20) var feat1_in: texture_2d<f32>;
@group(0) @binding(21) var restored_rgba_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(22) var restored_rgba_in: texture_2d<f32>;

fn luma709(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// -----------------------------------------------------------------------------
// Pass 1A: NV12 (Rec. 709 limited) -> RGBA8 full-resolution (1920x1080)
// -----------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn nv12_to_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let uv_coord = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(f32(u.width), f32(u.height));
    let y_raw = textureLoad(y_tex, vec2<i32>(gid.xy), 0).r * 255.0;
    let uv_raw = textureSampleLevel(uv_tex, lin_sampler, uv_coord, 0.0).rg * 255.0;

    let y = (y_raw - 16.0) / 219.0;
    let cb = (uv_raw.x - 128.0) / 224.0;
    let cr = (uv_raw.y - 128.0) / 224.0;

    let r = clamp(y + 1.5748 * cr, 0.0, 1.0);
    let g = clamp(y - 0.1873 * cb - 0.4681 * cr, 0.0, 1.0);
    let b = clamp(y + 1.8556 * cb, 0.0, 1.0);

    textureStore(src_rgba_out, vec2<i32>(gid.xy), vec4<f32>(r, g, b, 1.0));
}

// -----------------------------------------------------------------------------
// Pass 1B: Ultra-Light Micro-Noise Stabilizer (avoids double-blur with aurora-camera-denoise)
// Only settles tiny static grain (< 0.025); any real facial expression or motion uses 100% live frame.
// -----------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn temporal_denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let curr = textureLoad(src_rgba_in, p, 0).rgb;

    if (u.frame_index == 0u) {
        textureStore(clean_rgba_out, p, vec4<f32>(curr, 1.0));
        return;
    }

    let prev = textureLoad(prev_clean_in, p, 0).rgb;
    let diff = length(curr - prev);
    let static_conf = 1.0 - smoothstep(0.008, 0.032, diff);
    let denoised = mix(curr, prev, static_conf * 0.16);

    textureStore(clean_rgba_out, p, vec4<f32>(denoised, 1.0));
}

// -----------------------------------------------------------------------------
// Pass 1C..1F: 2-Iteration GPU Richardson-Lucy Optical Deconvolution (Rgba16Float)
// Mathematically inverts the ~1.8px Gaussian optical/ISP softness of Linux apple-isp,
// bringing soft eyes, eyelashes, beard, and hair back into crisp focus without harsh halos.
// -----------------------------------------------------------------------------
fn gaussian9_clean(uv: vec2<f32>, step_uv: vec2<f32>) -> vec3<f32> {
    let c  = textureSampleLevel(clean_rgba_in, lin_sampler, uv, 0.0).rgb;
    let n  = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>( 0.0,       -step_uv.y), 0.0).rgb;
    let s  = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>( 0.0,        step_uv.y), 0.0).rgb;
    let w  = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>(-step_uv.x,  0.0),       0.0).rgb;
    let e  = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>( step_uv.x,  0.0),       0.0).rgb;
    let nw = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>(-step_uv.x, -step_uv.y), 0.0).rgb;
    let ne = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>( step_uv.x, -step_uv.y), 0.0).rgb;
    let sw = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>(-step_uv.x,  step_uv.y), 0.0).rgb;
    let se = textureSampleLevel(clean_rgba_in, lin_sampler, uv + vec2<f32>( step_uv.x,  step_uv.y), 0.0).rgb;
    return c * 0.25 + (n + s + w + e) * 0.125 + (nw + ne + sw + se) * 0.0625;
}

fn gaussian9_feat0(uv: vec2<f32>, step_uv: vec2<f32>) -> vec3<f32> {
    let c  = textureSampleLevel(feat0_in, lin_sampler, uv, 0.0).rgb;
    let n  = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>( 0.0,       -step_uv.y), 0.0).rgb;
    let s  = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>( 0.0,        step_uv.y), 0.0).rgb;
    let w  = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>(-step_uv.x,  0.0),       0.0).rgb;
    let e  = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>( step_uv.x,  0.0),       0.0).rgb;
    let nw = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>(-step_uv.x, -step_uv.y), 0.0).rgb;
    let ne = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>( step_uv.x, -step_uv.y), 0.0).rgb;
    let sw = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>(-step_uv.x,  step_uv.y), 0.0).rgb;
    let se = textureSampleLevel(feat0_in, lin_sampler, uv + vec2<f32>( step_uv.x,  step_uv.y), 0.0).rgb;
    return c * 0.25 + (n + s + w + e) * 0.125 + (nw + ne + sw + se) * 0.0625;
}

fn gaussian9_feat1(uv: vec2<f32>, step_uv: vec2<f32>) -> vec3<f32> {
    let c  = textureSampleLevel(feat1_in, lin_sampler, uv, 0.0).rgb;
    let n  = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>( 0.0,       -step_uv.y), 0.0).rgb;
    let s  = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>( 0.0,        step_uv.y), 0.0).rgb;
    let w  = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>(-step_uv.x,  0.0),       0.0).rgb;
    let e  = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>( step_uv.x,  0.0),       0.0).rgb;
    let nw = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>(-step_uv.x, -step_uv.y), 0.0).rgb;
    let ne = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>( step_uv.x, -step_uv.y), 0.0).rgb;
    let sw = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>(-step_uv.x,  step_uv.y), 0.0).rgb;
    let se = textureSampleLevel(feat1_in, lin_sampler, uv + vec2<f32>( step_uv.x,  step_uv.y), 0.0).rgb;
    return c * 0.25 + (n + s + w + e) * 0.125 + (nw + ne + sw + se) * 0.0625;
}

// Pass 1C: Iteration 1 — compute ratio I_0 / (I_0 * G_1.85) into feat0 (HDR Rgba16Float)
@compute @workgroup_size(16, 16)
fn rl_ratio1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let inv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height));
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * inv;
    let orig = textureLoad(clean_rgba_in, vec2<i32>(gid.xy), 0).rgb;
    let blur = gaussian9_clean(uv, inv * 1.85);
    let ratio = clamp((orig + vec3<f32>(0.02)) / (blur + vec3<f32>(0.02)), vec3<f32>(0.70), vec3<f32>(1.42));
    // Store local edge energy in .a so flat noisy patches don't amplify grain
    let edge_energy = length(orig - blur);
    textureStore(feat0_out, vec2<i32>(gid.xy), vec4<f32>(ratio, edge_energy));
}

// Pass 1D: Iteration 1 — update I_1 = I_0 * (ratio1 * G_1.85) into feat1 (Rgba16Float)
@compute @workgroup_size(16, 16)
fn rl_update1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let inv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height));
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * inv;
    let orig = textureLoad(clean_rgba_in, vec2<i32>(gid.xy), 0).rgb;

    if (u.sharpness <= 0.01) {
        textureStore(feat1_out, vec2<i32>(gid.xy), vec4<f32>(orig, 1.0));
        return;
    }

    let corr = gaussian9_feat0(uv, inv * 1.85);
    let energy = textureLoad(feat0_in, vec2<i32>(gid.xy), 0).a;
    // Sensitive gate (0.012..0.045) passes soft facial details while suppressing flat sensor grain
    let gate = smoothstep(0.010, 0.042, energy);
    let factor = mix(vec3<f32>(1.0), corr, u.sharpness * (0.35 + 0.65 * gate));
    let i1 = clamp(orig * factor, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(feat1_out, vec2<i32>(gid.xy), vec4<f32>(i1, energy));
}

// Pass 1E: Iteration 2 — compute ratio I_0 / (I_1 * G_1.45) into feat0 (HDR Rgba16Float)
@compute @workgroup_size(16, 16)
fn rl_ratio2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let inv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height));
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * inv;
    let orig = textureLoad(clean_rgba_in, vec2<i32>(gid.xy), 0).rgb;
    let blur = gaussian9_feat1(uv, inv * 1.45);
    let ratio = clamp((orig + vec3<f32>(0.02)) / (blur + vec3<f32>(0.02)), vec3<f32>(0.76), vec3<f32>(1.32));
    let energy = textureLoad(feat1_in, vec2<i32>(gid.xy), 0).a;
    textureStore(feat0_out, vec2<i32>(gid.xy), vec4<f32>(ratio, energy));
}

// Pass 1F: Iteration 2 — update I_2 = I_1 * (ratio2 * G_1.45) into restored_rgba (Rgba8Unorm)
@compute @workgroup_size(16, 16)
fn rl_update2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let inv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height));
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * inv;
    let i1_pack = textureLoad(feat1_in, vec2<i32>(gid.xy), 0);
    let i1 = i1_pack.rgb;

    if (u.sharpness <= 0.01) {
        textureStore(restored_rgba_out, vec2<i32>(gid.xy), vec4<f32>(i1, 1.0));
        return;
    }

    let corr = gaussian9_feat0(uv, inv * 1.45);
    let gate = smoothstep(0.010, 0.042, i1_pack.a);
    let factor = mix(vec3<f32>(1.0), corr, u.sharpness * 0.85 * (0.30 + 0.70 * gate));
    let i2 = clamp(i1 * factor, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(restored_rgba_out, vec2<i32>(gid.xy), vec4<f32>(i2, 1.0));
}

// -----------------------------------------------------------------------------
// Pass 2: Prepare half-res (960x540) foreground-masked background for bokeh
// -----------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn prepare_half_bg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hw = u.width / 2u;
    let hh = u.height / 2u;
    if (gid.x >= hw || gid.y >= hh) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(f32(hw), f32(hh));

    var color: vec3<f32>;
    var bg_weight: f32;

    if (u.bg_mode == 2u) {
        color = textureSampleLevel(bg_img_tex, lin_sampler, uv, 0.0).rgb;
        bg_weight = 1.0;
    } else {
        color = textureSampleLevel(clean_rgba_in, lin_sampler, uv, 0.0).rgb;
        let alpha = textureSampleLevel(alpha_tex, lin_sampler, uv, 0.0).r;
        let texel = vec2<f32>(1.0 / f32(hw), 1.0 / f32(hh)) * 1.5;
        let a_max = max(
            alpha,
            max(
                max(
                    textureSampleLevel(alpha_tex, lin_sampler, uv + vec2<f32>(texel.x, 0.0), 0.0).r,
                    textureSampleLevel(alpha_tex, lin_sampler, uv - vec2<f32>(texel.x, 0.0), 0.0).r
                ),
                max(
                    textureSampleLevel(alpha_tex, lin_sampler, uv + vec2<f32>(0.0, texel.y), 0.0).r,
                    textureSampleLevel(alpha_tex, lin_sampler, uv - vec2<f32>(0.0, texel.y), 0.0).r
                )
            )
        );
        bg_weight = smoothstep(0.08, 0.90, 1.0 - a_max);
    }

    textureStore(half_bg_out, vec2<i32>(gid.xy), vec4<f32>(color, bg_weight));
}

// -----------------------------------------------------------------------------
// Pass 3: Optical Golden-Angle Disk Bokeh (half-res 960x540, 48 taps)
// -----------------------------------------------------------------------------
const GOLDEN_ANGLE: f32 = 2.39996323;
const BOKEH_TAPS: u32 = 48u;

@compute @workgroup_size(16, 16)
fn bokeh_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hw = u.width / 2u;
    let hh = u.height / 2u;
    if (gid.x >= hw || gid.y >= hh) {
        return;
    }

    let strength = select(u.blur_strength, u.bg_image_blur, u.bg_mode == 2u);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(f32(hw), f32(hh));

    if (strength <= 0.01) {
        let c = textureSampleLevel(half_bg_in, lin_sampler, uv, 0.0).rgb;
        textureStore(bokeh_out, vec2<i32>(gid.xy), vec4<f32>(c, 1.0));
        return;
    }

    let radius_px = 3.0 + strength * 23.0;
    let inv_size = vec2<f32>(1.0 / f32(hw), 1.0 / f32(hh));

    let hash = fract(sin(dot(vec2<f32>(gid.xy), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let base_angle = hash * 6.2831853;

    var acc_masked = vec3<f32>(0.0);
    var w_masked = 0.0;

    var acc_fallback = vec3<f32>(0.0);
    var w_fallback = 0.0;

    for (var i: u32 = 0u; i < BOKEH_TAPS; i = i + 1u) {
        let frac = (f32(i) + 0.5) / f32(BOKEH_TAPS);
        let r = sqrt(frac) * radius_px;
        let theta = base_angle + f32(i) * GOLDEN_ANGLE;
        let offset = vec2<f32>(cos(theta), sin(theta)) * r * inv_size;

        let s = textureSampleLevel(
            half_bg_in,
            lin_sampler,
            clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0)),
            0.0
        );
        let luma = luma709(s.rgb);
        let bokeh_boost = 1.0 + luma * luma * luma * 1.4;

        let wm = s.a * bokeh_boost;
        acc_masked += s.rgb * wm;
        w_masked += wm;

        acc_fallback += s.rgb * bokeh_boost;
        w_fallback += bokeh_boost;
    }

    let fallback_col = acc_fallback / max(w_fallback, 1e-4);
    let masked_col = acc_masked / max(w_masked, 1e-4);
    let mask_confidence = smoothstep(0.05, 1.5, w_masked);
    let final_col = mix(fallback_col, masked_col, mask_confidence);

    textureStore(bokeh_out, vec2<i32>(gid.xy), vec4<f32>(final_col, 1.0));
}

// -----------------------------------------------------------------------------
// Pass 4: Center Stage Bicubic Zoom + 768x432 Edge-Guided Matting + RVM Decontam +
//         3-Band Apple-ISP Clarity & CAS + Studio Light + Rec. 709 NV12 Packer
// -----------------------------------------------------------------------------
fn sample_restored_crisp(uv: vec2<f32>) -> vec3<f32> {
    let tex_size = vec2<f32>(f32(u.width), f32(u.height));
    let inv_size = vec2<f32>(1.0) / tex_size;

    let sample_pos = uv * tex_size;
    let tc1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tc1;
    let f2 = f * f;
    let f3 = f2 * f;

    let w0 = f2 - 0.5 * (f3 + f);
    let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    let w3 = 0.5 * (f3 - f2);
    let w2 = 1.0 - w0 - w1 - w3;

    let w12 = w1 + w2;
    let tc0 = (tc1 - 1.0) * inv_size;
    let tc3 = (tc1 + 2.0) * inv_size;
    let tc12 = (tc1 + w2 / w12) * inv_size;

    let c0 = textureSampleLevel(restored_rgba_in, lin_sampler, vec2<f32>(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y);
    let c1 = textureSampleLevel(restored_rgba_in, lin_sampler, vec2<f32>(tc0.x,  tc12.y), 0.0).rgb * (w0.x  * w12.y);
    let c2 = textureSampleLevel(restored_rgba_in, lin_sampler, vec2<f32>(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y);
    let c3 = textureSampleLevel(restored_rgba_in, lin_sampler, vec2<f32>(tc3.x,  tc12.y), 0.0).rgb * (w3.x  * w12.y);
    let c4 = textureSampleLevel(restored_rgba_in, lin_sampler, vec2<f32>(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);

    let total_w = (w12.x * w0.y) + (w0.x * w12.y) + (w12.x * w12.y) + (w3.x * w12.y) + (w12.x * w3.y);
    return clamp((c0 + c1 + c2 + c3 + c4) / max(total_w, 1e-4), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_sensor_restored(src_uv: vec2<f32>) -> vec3<f32> {
    if (u.crop.z <= 1.005) {
        let p = vec2<i32>(
            clamp(i32(src_uv.x * f32(u.width)), 0, i32(u.width) - 1),
            clamp(i32(src_uv.y * f32(u.height)), 0, i32(u.height) - 1)
        );
        return textureLoad(restored_rgba_in, p, 0).rgb;
    }
    return sample_restored_crisp(src_uv);
}

fn sample_guided_alpha(src_uv: vec2<f32>, center_rgb: vec3<f32>) -> f32 {
    let c_alpha = textureSampleLevel(alpha_tex, lin_sampler, src_uv, 0.0).r;
    if (c_alpha < 0.02 || c_alpha > 0.98) {
        return smoothstep(0.025, 0.965, c_alpha);
    }

    let step_uv = (vec2<f32>(1.0) / max(u.mask_size, vec2<f32>(256.0))) * 0.85;
    let inv_color_sigma2 = 1.0 / (0.10 * 0.10);

    var acc = c_alpha * 1.5;
    var wsum = 1.5;

    let offsets = array<vec2<f32>, 4>(
        vec2<f32>( step_uv.x, 0.0),
        vec2<f32>(-step_uv.x, 0.0),
        vec2<f32>(0.0,  step_uv.y),
        vec2<f32>(0.0, -step_uv.y)
    );

    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let nuv = clamp(src_uv + offsets[k], vec2<f32>(0.0), vec2<f32>(1.0));
        let n_rgb = textureSampleLevel(clean_rgba_in, lin_sampler, nuv, 0.0).rgb;
        let n_a = textureSampleLevel(alpha_tex, lin_sampler, nuv, 0.0).r;
        let diff = n_rgb - center_rgb;
        let w = exp(-dot(diff, diff) * inv_color_sigma2);
        acc += n_a * w;
        wsum += w;
    }

    let refined = acc / max(wsum, 1e-4);
    return smoothstep(0.025, 0.965, refined);
}

/// 3-Band Apple-ISP Polish on top of 2-Iteration Richardson-Lucy Optical Deconvolution:
/// - Band 1: 1px FidelityFX CAS (fine micro-crispness)
/// - Band 2: 2.8px optical detail contrast
/// - Band 3: 6.0px local Clarity / de-haze (matches Apple ISP tone pop)
fn polish_foreground(src_uv: vec2<f32>, center: vec3<f32>) -> vec3<f32> {
    let s = u.sharpness;
    if (s <= 0.01) {
        return center;
    }

    let step_uv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height)) / max(u.crop.z, 1.0);

    // Band 1: FidelityFX CAS (1px)
    let a = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-step_uv.x, -step_uv.y), 0.0).rgb;
    let b = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( 0.0,       -step_uv.y), 0.0).rgb;
    let c = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( step_uv.x, -step_uv.y), 0.0).rgb;
    let d = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-step_uv.x,  0.0),       0.0).rgb;
    let f = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( step_uv.x,  0.0),       0.0).rgb;
    let g = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-step_uv.x,  step_uv.y), 0.0).rgb;
    let h = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( 0.0,        step_uv.y), 0.0).rgb;
    let i = textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( step_uv.x,  step_uv.y), 0.0).rgb;

    let mn_cross = min(min(min(d, center), min(f, b)), h);
    let mn_diag  = min(min(a, c), min(g, i));
    let mn_rgb   = (mn_cross + min(mn_cross, mn_diag)) * 0.5;

    let mx_cross = max(max(max(d, center), max(f, b)), h);
    let mx_diag  = max(max(a, c), max(g, i));
    let mx_rgb   = (mx_cross + max(mx_cross, mx_diag)) * 0.5;

    let rcp_mx = vec3<f32>(1.0) / max(mx_rgb, vec3<f32>(1e-4));
    let amp = sqrt(clamp(min(mn_rgb, vec3<f32>(1.0) - mx_rgb) * rcp_mx, vec3<f32>(0.0), vec3<f32>(1.0)));
    let peak = -mix(0.0, 0.195, s);
    let w_cas = amp * peak;

    let cas_rgb = clamp(
        (center + (b + d + f + h) * w_cas) / (vec3<f32>(1.0) + 4.0 * w_cas),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    // Band 2: Mid-radius optical detail (2.8px)
    let r2 = step_uv * 2.8;
    let blur_mid = (
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( r2.x,  r2.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-r2.x,  r2.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( r2.x, -r2.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-r2.x, -r2.y), 0.0).rgb
    ) * 0.25;
    let detail_mid = clamp(cas_rgb - blur_mid, vec3<f32>(-0.14), vec3<f32>(0.14));

    // Band 3: Broad local clarity (6.0px) — restores Apple ISP micro-contrast across facial features
    let r3 = step_uv * 6.0;
    let blur_wide = (
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( r3.x,  0.0),  0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-r3.x,  0.0),  0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( 0.0,   r3.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( 0.0,  -r3.y), 0.0).rgb
    ) * 0.25;
    let detail_wide = clamp(cas_rgb - blur_wide, vec3<f32>(-0.12), vec3<f32>(0.12));

    var out_c = clamp(
        cas_rgb + detail_mid * (0.42 * s) + detail_wide * (0.28 * s),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    // Black-point de-haze & gentle filmic S-curve
    let black_floor = 0.038 * s;
    out_c = clamp((out_c - vec3<f32>(black_floor)) / (1.0 - black_floor), vec3<f32>(0.0), vec3<f32>(1.0));
    let s_curve = out_c * out_c * (3.0 - 2.0 * out_c);
    out_c = mix(out_c, s_curve, 0.20 * s);

    let lum = luma709(out_c);
    out_c = clamp(mix(vec3<f32>(lum), out_c, 1.0 + 0.08 * s), vec3<f32>(0.0), vec3<f32>(1.0));
    return out_c;
}

fn eval_pixel(px: u32, py: u32) -> vec3<f32> {
    let inv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height));
    let out_uv = (vec2<f32>(f32(px), f32(py)) + vec2<f32>(0.5)) * inv;

    let zoom = max(u.crop.z, 1.0);
    let src_uv = clamp(u.crop.xy + (out_uv - vec2<f32>(0.5)) / zoom, vec2<f32>(0.0), vec2<f32>(1.0));

    let clean_bg_raw = textureSampleLevel(clean_rgba_in, lin_sampler, src_uv, 0.0).rgb;
    let restored_fg = sample_sensor_restored(src_uv);
    let alpha = sample_guided_alpha(src_uv, clean_bg_raw);

    var fg = restored_fg;
    if (alpha > 0.02 && u.sharpness > 0.01) {
        fg = polish_foreground(src_uv, restored_fg);
    }

    // RVM `fgr` edge color decontamination
    if (u.crop.w > 0.5 && u.bg_mode != 0u && alpha > 0.03 && alpha < 0.92) {
        let edge_band = 4.0 * alpha * (1.0 - alpha);
        let fgr_clean = textureSampleLevel(fgr_tex, lin_sampler, src_uv, 0.0).rgb;
        let decontam_strength = select(0.45, 0.74, u.bg_mode == 2u || u.bg_mode == 3u);
        fg = mix(fg, fgr_clean, edge_band * decontam_strength);
    }

    var bg: vec3<f32>;
    if (u.bg_mode == 1u) {
        let bokeh = textureSampleLevel(bokeh_in, lin_sampler, src_uv, 0.0).rgb;
        bg = mix(clean_bg_raw, bokeh, smoothstep(0.0, 0.25, u.blur_strength));
    } else if (u.bg_mode == 2u) {
        let bg_zoom = 1.0 + (zoom - 1.0) * 0.25;
        let bg_center = mix(vec2<f32>(0.5), u.crop.xy, 0.25);
        let bg_uv = clamp(bg_center + (out_uv - vec2<f32>(0.5)) / bg_zoom, vec2<f32>(0.0), vec2<f32>(1.0));
        if (u.bg_image_blur > 0.01) {
            bg = textureSampleLevel(bokeh_in, lin_sampler, bg_uv, 0.0).rgb;
        } else {
            bg = textureSampleLevel(bg_img_tex, lin_sampler, bg_uv, 0.0).rgb;
        }
    } else if (u.bg_mode == 3u) {
        let d = (out_uv - vec2<f32>(0.5, 0.42)) * vec2<f32>(1.35, 1.0);
        let grad = clamp(1.12 - dot(d, d) * 0.55, 0.55, 1.15);
        bg = clamp(u.bg_color.rgb * grad, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        bg = fg;
    }

    if (u.studio_light > 0.01) {
        let sl = u.studio_light;
        let delta = (src_uv - u.centroid) * vec2<f32>(1.65, 1.0) * zoom;
        let dist2 = dot(delta, delta);

        let bg_vignette = clamp(1.0 - dist2 * 0.48, 0.52, 1.0);
        let bg_dim = mix(1.0, 0.60 * bg_vignette, sl);
        bg = bg * bg_dim;

        let key_falloff = exp(-dist2 * 2.4);
        let gamma_lift = 1.0 - sl * (0.06 + 0.16 * key_falloff);
        let lifted = pow(max(fg, vec3<f32>(1e-5)), vec3<f32>(gamma_lift));
        let exposure_gain = 1.0 + sl * 0.12 * key_falloff;
        let exposed = lifted * exposure_gain;
        let soft_clipped = exposed / (vec3<f32>(1.0) + max(exposed - vec3<f32>(0.88), vec3<f32>(0.0)) * 0.9);
        let warm_tint = mix(vec3<f32>(1.0), vec3<f32>(1.025, 1.006, 0.980), sl * key_falloff);
        fg = clamp(soft_clipped * warm_tint, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return clamp(mix(bg, fg, alpha), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rgb_to_y(c: vec3<f32>) -> u32 {
    let y = 16.0 + 219.0 * (0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b);
    return u32(clamp(round(y), 16.0, 235.0));
}

fn rgb_to_uv(c: vec3<f32>) -> vec2<u32> {
    let u_val = 128.0 + 224.0 * (-0.1146 * c.r - 0.3854 * c.g + 0.5000 * c.b);
    let v_val = 128.0 + 224.0 * ( 0.5000 * c.r - 0.4542 * c.g - 0.0458 * c.b);
    return vec2<u32>(
        u32(clamp(round(u_val), 16.0, 240.0)),
        u32(clamp(round(v_val), 16.0, 240.0))
    );
}

@compute @workgroup_size(16, 16)
fn compose_nv12(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words_per_row = u.width / 4u;
    let uv_rows = u.height / 2u;
    if (gid.x >= words_per_row || gid.y >= uv_rows) {
        return;
    }

    let x0 = gid.x * 4u;
    let y0 = gid.y * 2u;

    let c00 = eval_pixel(x0 + 0u, y0 + 0u);
    let c10 = eval_pixel(x0 + 1u, y0 + 0u);
    let c20 = eval_pixel(x0 + 2u, y0 + 0u);
    let c30 = eval_pixel(x0 + 3u, y0 + 0u);

    let c01 = eval_pixel(x0 + 0u, y0 + 1u);
    let c11 = eval_pixel(x0 + 1u, y0 + 1u);
    let c21 = eval_pixel(x0 + 2u, y0 + 1u);
    let c31 = eval_pixel(x0 + 3u, y0 + 1u);

    let y_row0 = rgb_to_y(c00)
        | (rgb_to_y(c10) << 8u)
        | (rgb_to_y(c20) << 16u)
        | (rgb_to_y(c30) << 24u);
    let y_row1 = rgb_to_y(c01)
        | (rgb_to_y(c11) << 8u)
        | (rgb_to_y(c21) << 16u)
        | (rgb_to_y(c31) << 24u);

    let uv_left = rgb_to_uv((c00 + c10 + c01 + c11) * 0.25);
    let uv_right = rgb_to_uv((c20 + c30 + c21 + c31) * 0.25);
    let uv_word = uv_left.x
        | (uv_left.y << 8u)
        | (uv_right.x << 16u)
        | (uv_right.y << 24u);

    nv12_out[y0 * words_per_row + gid.x] = y_row0;
    nv12_out[(y0 + 1u) * words_per_row + gid.x] = y_row1;

    let uv_base_words = (u.width * u.height) / 4u;
    nv12_out[uv_base_words + gid.y * words_per_row + gid.x] = uv_word;
}
