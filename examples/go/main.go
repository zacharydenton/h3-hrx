// libh3pipe from Go through cgo: prompt -> frames + samples, written as <out>.rgb and <out>.wav.
//   go build -o minimal . && ./minimal "A red fox ..." [frames] [steps] [out]     (from the repository root, or set H3_ROOT)
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../host
#cgo LDFLAGS: -L${SRCDIR}/../../build -lh3pipe -Wl,-rpath,${SRCDIR}/../../build
#include <stdlib.h>
#include "h3pipe.h"
#include "h3tok.h"

int goProgress(void *user, int step, int steps, double seconds);   // exported from Go below
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"unsafe"
)

//export goProgress
func goProgress(user unsafe.Pointer, step C.int, steps C.int, seconds C.double) C.int {
	fmt.Fprintf(os.Stderr, "  step %d/%d  %.1f s\n", int(step), int(steps), float64(seconds))
	return 0 // nonzero cancels
}

func fail(what string, err []C.char) {
	fmt.Fprintf(os.Stderr, "%s: %s\n", what, C.GoString(&err[0]))
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: minimal \"prompt\" [frames] [steps] [out]")
		os.Exit(64)
	}
	root, home, out := ".", ".", "minimal"
	if v := os.Getenv("H3_ROOT"); v != "" { root = v }
	if v := os.Getenv("HOME"); v != "" { home = v }
	if len(os.Args) > 4 { out = os.Args[4] }
	atoi := func(i int, dflt int) int { if len(os.Args) > i { v, _ := strconv.Atoi(os.Args[i]); return v }; return dflt }
	err := make([]C.char, 4096)
	if uint32(C.h3pipe_abi_version()) != uint32(C.H3PIPE_ABI_VERSION) { fmt.Fprintln(os.Stderr, "libh3pipe ABI mismatch"); os.Exit(1) }

	// text -> ids
	tok := C.h3tok_create(nil, &err[0], C.size_t(len(err)))   // the tokenizer compiled into libh3pipe
	if tok == nil { fail("tokenizer", err) }
	prompt := C.CString(os.Args[1]); defer C.free(unsafe.Pointer(prompt))
	ids := make([]C.int32_t, 4096)
	n := int(C.h3tok_encode(tok, prompt, &ids[0], C.size_t(len(ids))))
	C.h3tok_destroy(tok)
	if n < 0 || n > len(ids) { fmt.Fprintln(os.Stderr, "cannot tokenize the prompt"); os.Exit(1) }

	// a session
	cs := func(s string) *C.char { return C.CString(s) }
	loom := "loom-compile"
	if v := os.Getenv("LOOM_COMPILE"); v != "" { loom = v }
	models := os.Getenv("H3_MODELS"); if models == "" { models = home + "/comfy-models" }
	cfg := C.h3pipe_config{dit_file: cs(models + "/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"), te_file: cs(models + "/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
		video_vae_file: cs(models + "/vae/minimax_h3_video_vae_fp16.safetensors"), audio_vae_file: cs(models + "/vae/minimax_h3_audio_vae_fp32.safetensors"),
		kernel_sources: cs(root + "/kernels"), cache_dir: cs(root + "/build/kernel_cache"), loom_compile: cs(loom), attn_qk_bits: 8}
	var s *C.h3pipe_session
	if C.h3pipe_create(&cfg, &s, &err[0], C.size_t(len(err))) != 0 { fail("create", err) }

	// sizes from the parameters
	p := C.h3pipe_params{height: 480, width: 864, frames: C.int(atoi(2, 124)), steps: C.int(atoi(3, 31)), seed: 0, sampler: 1}
	var sh C.h3pipe_shape
	C.h3pipe_shape_for(&p, &sh)
	video := make([]C.float, 24*int(sh.latent_t)*int(sh.lat_h)*int(sh.lat_w))
	audio := make([]C.float, 64*int(sh.audio_t))
	fmt.Fprintf(os.Stderr, "%d frames, %dx%dx%d latents, %d audio latents, %d prompt tokens\n", int(sh.frames), int(sh.latent_t), int(sh.lat_h), int(sh.lat_w), int(sh.audio_t), n)
	if C.h3pipe_denoise(s, &ids[0], C.int(n), &p, nil, nil, &video[0], C.size_t(len(video)), &audio[0], C.size_t(len(audio)), C.h3pipe_progress(C.goProgress), nil, &err[0], C.size_t(len(err))) != 0 { fail("denoise", err) }

	frames := make([]byte, int(sh.frames)*int(p.height)*int(p.width)*3)
	samples := make([]C.float, 1600*int(sh.audio_t))
	if C.h3pipe_decode_video(s, &p, &video[0], C.size_t(len(video)), (*C.uint8_t)(unsafe.Pointer(&frames[0])), C.size_t(len(frames)), &err[0], C.size_t(len(err))) != 0 { fail("decode video", err) }
	if C.h3pipe_decode_audio(s, &audio[0], C.size_t(len(audio)), sh.audio_t, &samples[0], C.size_t(len(samples)), &err[0], C.size_t(len(err))) != 0 { fail("decode audio", err) }
	C.h3pipe_destroy(s)

	if e := os.WriteFile(out+".rgb", frames, 0o644); e != nil { panic(e) }
	f, e := os.Create(out + ".wav"); if e != nil { panic(e) }
	defer f.Close()
	nsamp := uint32(sh.audio_t) * 800
	hdr := []any{[]byte("RIFF"), uint32(36 + nsamp*4), []byte("WAVEfmt "), uint32(16), uint16(1), uint16(2), uint32(32000), uint32(128000), uint16(4), uint16(16), []byte("data"), uint32(nsamp * 4)}
	for _, h := range hdr { binary.Write(f, binary.LittleEndian, h) }
	pcm := make([]int16, 2*nsamp)
	for i := uint32(0); i < nsamp; i++ {
		for ch := 0; ch < 2; ch++ {
			v := float64(samples[uint32(ch)*nsamp+i]); if v < -1 { v = -1 }; if v > 1 { v = 1 }
			pcm[2*i+uint32(ch)] = int16(v * 32767)
		}
	}
	binary.Write(f, binary.LittleEndian, pcm)
	fmt.Fprintf(os.Stderr, "wrote %s.rgb (%d x %dx%d rgb24) and %s.wav\n", out, int(sh.frames), int(p.width), int(p.height), out)
}
