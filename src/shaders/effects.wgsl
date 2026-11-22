struct Uniforms {
    width: u32,
    height: u32,
    bg_mode: u32,        // 0 = None, 1 = Blur, 2 = Image, 3 = Color
    frame_index: u32,    // 0 resets temporal denoiser history
    blur_strength: f32,  // 0.0..1.0
    studio_light: f32,   // 0.0..1.0
    bg_image_blur: f32,  // 0.0..1.0
    sharpness: f32,      // 0.0..1.0 (Temporal Denoise + Residual CNN Restore + FidelityFX CAS + De-haze)
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
// Temporal Denoiser + 2-Stage Residual Restoration CNN textures
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
// Pass 1B: Motion-Adaptive Temporal GPU Denoiser (1920x1080, ~0.4 ms)
// Accumulates previous clean frame on static/slow regions, instantly resets on motion.
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

    // Compare both center pixel AND 3x3 neighborhood mean so real motion rejects history
    // immediately while zero-mean random sensor grain is smoothly averaged away.
    let xm = max(p.x - 1, 0);
    let xp = min(p.x + 1, i32(u.width) - 1);
    let ym = max(p.y - 1, 0);
    let yp = min(p.y + 1, i32(u.height) - 1);

    let curr_avg = (
        curr +
        textureLoad(src_rgba_in, vec2<i32>(xm, p.y), 0).rgb +
        textureLoad(src_rgba_in, vec2<i32>(xp, p.y), 0).rgb +
        textureLoad(src_rgba_in, vec2<i32>(p.x, ym), 0).rgb +
        textureLoad(src_rgba_in, vec2<i32>(p.x, yp), 0).rgb
    ) * 0.2;

    let prev_avg = (
        prev +
        textureLoad(prev_clean_in, vec2<i32>(xm, p.y), 0).rgb +
        textureLoad(prev_clean_in, vec2<i32>(xp, p.y), 0).rgb +
        textureLoad(prev_clean_in, vec2<i32>(p.x, ym), 0).rgb +
        textureLoad(prev_clean_in, vec2<i32>(p.x, yp), 0).rgb
    ) * 0.2;

    let diff_center = length(curr - prev);
    let diff_region = length(curr_avg - prev_avg);
    let motion = max(diff_center * 0.65, diff_region * 1.45);

    // Zero ghosting: history drops to 0.0 cleanly above motion threshold ~0.075
    let static_conf = 1.0 - smoothstep(0.014, 0.075, motion);
    let blend = static_conf * (0.28 + 0.18 * u.sharpness);
    let denoised = mix(curr, prev, blend);

    textureStore(clean_rgba_out, p, vec4<f32>(denoised, 1.0));
}

// -----------------------------------------------------------------------------
// Pass 1C: Residual Restoration CNN Layer 1 (8-channel 3x3 Feature Extractor, Rgba16Float)
// Extracts oriented fine-edge & micro-texture feature maps across 4 orientations + multi-scale structure.
// -----------------------------------------------------------------------------
fn load_clean(ix: i32, iy: i32) -> vec3<f32> {
    let cx = clamp(ix, 0, i32(u.width) - 1);
    let cy = clamp(iy, 0, i32(u.height) - 1);
    return textureLoad(clean_rgba_in, vec2<i32>(cx, cy), 0).rgb;
}

@compute @workgroup_size(16, 16)
fn cnn_layer1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let ix = i32(gid.x);
    let iy = i32(gid.y);

    // 3x3 luminance patch
    let l00 = luma709(load_clean(ix - 1, iy - 1));
    let l10 = luma709(load_clean(ix,     iy - 1));
    let l20 = luma709(load_clean(ix + 1, iy - 1));
    let l01 = luma709(load_clean(ix - 1, iy    ));
    let l11 = luma709(load_clean(ix,     iy    ));
    let l21 = luma709(load_clean(ix + 1, iy    ));
    let l02 = luma709(load_clean(ix - 1, iy + 1));
    let l12 = luma709(load_clean(ix,     iy + 1));
    let l22 = luma709(load_clean(ix + 1, iy + 1));

    // Oriented 3x3 filter bank (4 directional edge channels)
    let gx   = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
    let gy   = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
    let gd1  = (l20 + 2.0 * l10 + l21) - (l02 + 2.0 * l12 + l01);
    let gd2  = (l00 + 2.0 * l10 + l01) - (l22 + 2.0 * l12 + l21);

    // High-frequency ridge / fine detail kernels (horizontal, vertical, isotropic Laplacian)
    let ridge_h = l11 - 0.5 * (l01 + l21);
    let ridge_v = l11 - 0.5 * (l10 + l12);
    let lap     = l11 - 0.25 * (l10 + l01 + l21 + l12);
    let local_range = max(max(max(l00, l20), max(l02, l22)), max(max(l10, l12), max(l01, l21)))
                    - min(min(min(l00, l20), min(l02, l22)), min(min(l10, l12), min(l01, l21)));

    // Leaky/soft-thresholded activations suppress sub-noise-floor grain while passing real edges
    textureStore(feat0_out, vec2<i32>(ix, iy), vec4<f32>(abs(gx), abs(gy), abs(gd1), abs(gd2)));
    textureStore(feat1_out, vec2<i32>(ix, iy), vec4<f32>(ridge_h, ridge_v, lap, local_range));
}

// -----------------------------------------------------------------------------
// Pass 1D: Residual Restoration CNN Layer 2 (3x3 Spatial Feature Mixing -> Deblurred Residual)
// Coherently restores optical micro-edges (eyes, hair, beard) while gating isolated noise.
// -----------------------------------------------------------------------------
fn load_f0(ix: i32, iy: i32) -> vec4<f32> {
    let cx = clamp(ix, 0, i32(u.width) - 1);
    let cy = clamp(iy, 0, i32(u.height) - 1);
    return textureLoad(feat0_in, vec2<i32>(cx, cy), 0);
}

fn load_f1(ix: i32, iy: i32) -> vec4<f32> {
    let cx = clamp(ix, 0, i32(u.width) - 1);
    let cy = clamp(iy, 0, i32(u.height) - 1);
    return textureLoad(feat1_in, vec2<i32>(cx, cy), 0);
}

fn soft_shrink(x: f32, tau: f32) -> f32 {
    return sign(x) * max(abs(x) - tau, 0.0);
}

@compute @workgroup_size(16, 16)
fn cnn_layer2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.width || gid.y >= u.height) {
        return;
    }
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let base = textureLoad(clean_rgba_in, vec2<i32>(ix, iy), 0).rgb;

    if (u.sharpness <= 0.01) {
        textureStore(restored_rgba_out, vec2<i32>(ix, iy), vec4<f32>(base, 1.0));
        return;
    }

    // Spatial coherence across 3x3 feature maps: real edges are continuous along neighbors,
    // random sensor noise is isolated and fails the coherence gate.
    let f0_c = load_f0(ix, iy);
    let f0_n = load_f0(ix, iy - 1);
    let f0_s = load_f0(ix, iy + 1);
    let f0_w = load_f0(ix - 1, iy);
    let f0_e = load_f0(ix + 1, iy);

    let f1_c = load_f1(ix, iy);
    let f1_n = load_f1(ix, iy - 1);
    let f1_s = load_f1(ix, iy + 1);
    let f1_w = load_f1(ix - 1, iy);
    let f1_e = load_f1(ix + 1, iy);

    // Directional coherence score: vertical edges continue N/S (f0.x), horizontal edges continue W/E (f0.y)
    let coh_v = min(f0_c.x, 0.5 * (f0_n.x + f0_s.x));
    let coh_h = min(f0_c.y, 0.5 * (f0_w.y + f0_e.y));
    let coh_d = 0.5 * (f0_c.z + f0_c.w);
    let edge_coherence = smoothstep(0.035, 0.22, max(max(coh_v, coh_h), coh_d * 0.75));

    // Mix ridge & Laplacian residuals along coherent directions (soft-thresholded against grain)
    let tau = 0.008;
    let d_h = soft_shrink(f1_c.x * 0.65 + 0.175 * (f1_n.x + f1_s.x), tau);
    let d_v = soft_shrink(f1_c.y * 0.65 + 0.175 * (f1_w.y + f1_e.y), tau);
    let d_iso = soft_shrink(f1_c.z, tau * 1.2);

    // Anti-halo damping on ultra-high-contrast edges (prevents white outline on dark hair/glasses)
    let contrast_damp = 1.0 - smoothstep(0.32, 0.68, f1_c.w) * 0.55;

    let residual_luma = clamp(
        (d_h * 0.45 + d_v * 0.45 + d_iso * 0.55) * edge_coherence * contrast_damp * (1.25 * u.sharpness),
        -0.11,
        0.11
    );

    // Apply luma-preserving restoration residual
    let restored = clamp(base + vec3<f32>(residual_luma), vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(restored_rgba_out, vec2<i32>(ix, iy), vec4<f32>(restored, 1.0));
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
//         FidelityFX CAS + Clarity + De-Haze + Studio Light + Rec. 709 NV12 Packer
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

/// Joint Bilateral (Edge-Guided) alpha refinement adapted to dynamic mask_size (768x432 or 512x288).
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

/// Final foreground polish on top of Temporal Denoise + Residual CNN:
/// FidelityFX CAS + Local Clarity + Black-point de-haze.
fn polish_foreground(src_uv: vec2<f32>, center: vec3<f32>) -> vec3<f32> {
    let s = u.sharpness;
    if (s <= 0.01) {
        return center;
    }

    let step_uv = vec2<f32>(1.0 / f32(u.width), 1.0 / f32(u.height)) / max(u.crop.z, 1.0);

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
    let peak = -mix(0.0, 0.175, s);
    let w_cas = amp * peak;

    let cas_rgb = clamp(
        (center + (b + d + f + h) * w_cas) / (vec3<f32>(1.0) + 4.0 * w_cas),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    // Subtle mid-radius Clarity
    let r2 = step_uv * 2.5;
    let blur_mid = (
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( r2.x,  r2.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-r2.x,  r2.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>( r2.x, -r2.y), 0.0).rgb +
        textureSampleLevel(restored_rgba_in, lin_sampler, src_uv + vec2<f32>(-r2.x, -r2.y), 0.0).rgb
    ) * 0.25;
    let detail_mid = clamp(cas_rgb - blur_mid, vec3<f32>(-0.10), vec3<f32>(0.10));
    var out_c = clamp(cas_rgb + detail_mid * (0.32 * s), vec3<f32>(0.0), vec3<f32>(1.0));

    // Black-point de-haze & gentle filmic micro-contrast
    let black_floor = 0.032 * s;
    out_c = clamp((out_c - vec3<f32>(black_floor)) / (1.0 - black_floor), vec3<f32>(0.0), vec3<f32>(1.0));
    let s_curve = out_c * out_c * (3.0 - 2.0 * out_c);
    out_c = mix(out_c, s_curve, 0.16 * s);

    let lum = luma709(out_c);
    out_c = clamp(mix(vec3<f32>(lum), out_c, 1.0 + 0.07 * s), vec3<f32>(0.0), vec3<f32>(1.0));
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
