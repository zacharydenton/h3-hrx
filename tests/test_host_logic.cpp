// CPU-only checks: no runtime implementation or model files are linked/loaded.
#include "../host/h3pipe.cpp"
#include <cassert>

int main() {
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
    puts("PASS host layout, MRoPE, decoder shape and short-clip checks");
}
