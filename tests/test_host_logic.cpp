// CPU-only checks: no runtime implementation or model files are linked/loaded.
#include "../host/h3pipe.cpp"
#include <cassert>
#include <climits>

Rt &rt() { throw std::logic_error("no runtime in this test: every call here must be refused before it reaches the session"); }

int main() {
    // Check every half bit pattern, including signed zero and subnormals. F16C
    // may quiet signaling NaNs, but all finite values must be bit-identical.
    const HalfRow convert = half_row_converter();
    for (unsigned first = 0; first < 65536; first += 16) {
        uint16_t in[16]; float scalar[16], actual[16];
        for (unsigned i = 0; i < 16; ++i) in[i] = uint16_t(first + i);
        half_row_scalar(in, scalar); convert(in, actual);
        for (int i = 0; i < 16; ++i) {
            if (std::isnan(scalar[i])) assert(std::isnan(actual[i]));
            else assert(std::memcmp(scalar + i, actual + i, sizeof(float)) == 0);
            if (std::isfinite(scalar[i]))
                assert(unit_to_byte(scalar[i]) == uint8_t(std::lround(std::min(std::max(scalar[i], 0.0f), 1.0f) * 255.0f)));
        }
    }
    // Exercise both neighboring floats at every pixel rounding boundary.
    for (int i = 0; i < 255; ++i) {
        const float midpoint = float((double(i) + 0.5) / 255.0);
        for (float v : {std::nextafter(midpoint, 0.0f), midpoint, std::nextafter(midpoint, 1.0f)})
            assert(unit_to_byte(v) == uint8_t(std::lround(v * 255.0f)));
    }
    assert(gemm_m_group_for(37723, FFN, HID, 8) == 2);
    assert(vae_fast_m_group_for(1797, 8192, 2048) == 1);
    for (int n : {2048, 6144, 16384}) assert(vae_fast_m_group_for(1797, 2048, n) == 15);
    for (size_t tokens : {size_t(1), size_t(517), size_t(1796), size_t(1798)})
        assert(vae_fast_m_group_for(tokens, 2048, 16384) == m_group_for(tokens, 128));
    assert(vae_fast_m_group_for(1797, 512, 1024) == m_group_for(1797, 128));
    assert(gemm_m_group_for(32768, FFN, HID, 8) == 2);
    for (int bits : {4, 8, 16}) {
        assert(gemm_m_group_for(16000, FFN, HID, bits) == m_group_for(16000));
        assert(gemm_m_group_for(37723, HID, 3 * HEADS * HEAD_DIM, bits) == m_group_for(37723));
        if (bits != 8) assert(gemm_m_group_for(37723, FFN, HID, bits) == m_group_for(37723));
    }
    // The launch must cover the entire final row group selected at compile time,
    // including when the runtime row count would have selected a different group.
    for (size_t tokens : {1u, 255u, 256u, 257u, 513u, 16000u, 32767u, 32768u, 37723u}) {
        for (unsigned group : {1u, 2u, 3u, 4u, 15u}) {
            const size_t gy = gemm_grid_y(tokens, group);
            assert(gy % group == 0 && gy * 256 >= tokens);
            assert((gy - group) * 256 < tokens);
        }
    }
    int counts[3] = {};
    for (int j = 0; j < 64; ++j) ++counts[mrope_axis(j)];
    assert(counts[0] == 24 && counts[1] == 20 && counts[2] == 20);
    assert(mrope_axis(1) == 1 && mrope_axis(2) == 2);
    assert(mrope_axis(58) == 1 && mrope_axis(59) == 2 && mrope_axis(63) == 0);

    h3pipe_ref image{}; image.kind = 0; image.latent_t = 1; image.lat_h = image.lat_w = 4;
    h3pipe_ref audio{}; audio.kind = 1; audio.audio_t = 3;
    h3pipe_keyframe keyframe{};
    Layout layout(12, 7, 4, 6, 5, {image, audio}, {keyframe});
    layout.mark_vision(2, 4);
    layout.mark_vision(9, 2);
    for (int i = 0; i < 12; ++i) assert(layout.adaln_rows[i] == ((i >= 1 && i <= 6) || i >= 8 ? 0 : 1));
    bool rejected = false;
    try { layout.mark_vision(11, 2); } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    const size_t prefix = layout.text_len + layout.ref_rows;
    assert(std::find(layout.tclass.begin(), layout.tclass.begin() + prefix, 2) != layout.tclass.begin() + prefix);
    assert(std::find(layout.tclass.begin(), layout.tclass.begin() + prefix, 3) != layout.tclass.begin() + prefix);
    for (size_t i = prefix; i < layout.seq_len; ++i) assert(layout.tclass[i] >= 0 && layout.tclass[i] < 2);
    assert(layout.seq_len - prefix == layout.audio_rows + layout.video_rows);

    DecoderGrid grid{7, 4, 6};
    assert(grid.matches(7, 4, 6));
    assert(!grid.matches(7, 6, 4));       // equal token count, different rotary coordinates
    assert(!grid.matches(2, 14, 6));

    for (int frames : {5, 22, 39, 124}) {
        h3pipe_params p{}; p.frames = frames; p.height = 64; p.width = 96;
        h3pipe_shape shape{}; assert(h3pipe_shape_for(&p, &shape) == 0);
        const int padding = (-(shape.latent_t + VAE_TOKEN_DROP) % VAE_CHUNK + VAE_CHUNK) % VAE_CHUNK;
        const int chunks = decoder_chunks(shape.latent_t, padding);
        // Each full clip contributes 17 frames and the final 2-token tail contributes 5.
        const int decoded = shape.latent_t == 2 ? 2 * VAE_TRATIO - 3 : chunks * 17 + 5;
        assert(decoded == frames);
    }
    // the public shape call refuses what nothing downstream could allocate from
    for (auto bad : {std::tuple<int, int, int>{0, 0, 5}, {-32, 32, 5}, {31, 32, 5}, {32, 33, 5}, {32, 32, 0}, {32, 32, -1}, {32, 32, INT_MAX}, {MAX_SIDE + 32, 32, 5}}) {
        h3pipe_params p{}; std::tie(p.height, p.width, p.frames) = bad; h3pipe_shape shape{};
        assert(h3pipe_shape_for(&p, &shape) == H3PIPE_INVALID_ARGUMENT);
    }
    { h3pipe_params p{}; p.height = p.width = 32; p.frames = 1; h3pipe_shape shape{}; assert(h3pipe_shape_for(&p, &shape) == H3PIPE_OK && shape.frames == 5 && shape.latent_t == 2 && shape.lat_h == 2); }
    { h3pipe_params p{}; p.height = p.width = MAX_SIDE; p.frames = MAX_FRAMES; h3pipe_shape shape{}; assert(h3pipe_shape_for(&p, &shape) == H3PIPE_OK && shape.frames >= MAX_FRAMES && shape.audio_t > 0); }
    assert(h3pipe_shape_for(nullptr, nullptr) == H3PIPE_INVALID_ARGUMENT);
    // the entry points reject an undersized buffer before touching the session
    { char err[256]; h3pipe_params p{}; p.height = p.width = 64; p.frames = 22; float latents[1]; uint8_t frames[1]; float samples[1];
      assert(h3pipe_decode_video(reinterpret_cast<h3pipe_session *>(1), &p, latents, 1, frames, 1, err, sizeof err) == H3PIPE_INVALID_ARGUMENT && std::string(err).find("at least") != std::string::npos);
      assert(h3pipe_decode_audio(reinterpret_cast<h3pipe_session *>(1), latents, 64, 1, samples, 1599, err, sizeof err) == H3PIPE_INVALID_ARGUMENT);
      assert(h3pipe_text_in(reinterpret_cast<h3pipe_session *>(1), reinterpret_cast<const int32_t *>(latents), 1, samples, HID - 1, err, sizeof err) == H3PIPE_INVALID_ARGUMENT);
      p.height = 31; assert(h3pipe_denoise(reinterpret_cast<h3pipe_session *>(1), reinterpret_cast<const int32_t *>(latents), 1, &p, nullptr, nullptr, latents, 1, samples, 1, nullptr, nullptr, err, sizeof err) == H3PIPE_INVALID_ARGUMENT); }
    puts("PASS host layout, MRoPE, decoder shape, short-clip, shape validation and buffer checks");
}
