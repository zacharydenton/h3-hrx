// CPU-only checks of the h3 command's helpers, built with AddressSanitizer: the option table, the tokenizer buffer
// (h3tok_encode reports the count the text needs even past the buffer), output-location checks and the WAV writer.
#define main h3_cli_main
#include "../host/h3_cli.cpp"
#undef main
#include <cassert>
#include <fstream>

int main(int argc, char **argv) {
    assert(argc == 2); const std::string dir = argv[1];
    // a minimal byte-level vocabulary: every '0' is one token
    const std::string tok_json = dir + "/tokenizer.json";
    std::ofstream(tok_json) << R"({"model":{"type":"BPE","vocab":{"0":0},"merges":[]}})";
    Tok tok(tok_json); assert(tok.t);
    for (size_t n : {size_t(1), size_t(4096), size_t(4097), size_t(32769), size_t(70000)}) {
        std::vector<int32_t> ids; assert(tok.encode(std::string(n, '0'), &ids)); assert(ids.size() == n);
        for (int32_t id : ids) assert(id == 0);
    }

    // the option table: unknown options, missing values and values that look like flags
    auto parse = [](std::vector<const char *> args, Options *o, std::string *err) { args.insert(args.begin(), "h3"); return parse_options(int(args.size()), const_cast<char **>(args.data()), o, err); };
    { Options o; std::string err; assert(!parse({"--precison", "int8"}, &o, &err) && err.find("unknown option") != std::string::npos); }
    { Options o; std::string err; assert(!parse({"-p"}, &o, &err) && err.find("needs a value") != std::string::npos); }
    { Options o; std::string err; assert(parse({"-p", "--help", "ref.jpg", "--no-decode"}, &o, &err)); assert(!o.has("--help") && std::string(o.get("-p")) == "--help" && o.has("--no-decode") && o.files == std::vector<std::string>{"ref.jpg"}); }
    { Options o; std::string err; int v = -1;
      assert(parse({"--frames", "124", "--steps", "12x", "--width", "0"}, &o, &err));
      assert(parse_int(o, "--frames", 5, 1, 1 << 20, &v, &err) && v == 124);
      assert(!parse_int(o, "--steps", 31, 2, 1000, &v, &err) && err.find("--steps") != std::string::npos);
      assert(!parse_int(o, "--width", 864, 32, 8192, &v, &err));
      assert(parse_int(o, "--height", 480, 32, 8192, &v, &err) && v == 480);
      std::string c; assert(!parse_choice(o, "--precision", "int7", {"int8", "bf16", "int4"}, &c, &err) && err.find("--precision") != std::string::npos);
      assert(parse_choice(o, "--attn", "i8", {"f16", "i8", "i4"}, &c, &err) && c == "i8"); }

    // output locations are checked before the long run
    assert(dir_writable(dir + "/clip.mp4") && dir_writable("clip.mp4") && !dir_writable(dir + "/missing/clip.mp4"));
    assert(!write_file(dir + "/missing/x.bin", "x", 1));
    const uint32_t frames = 3; std::vector<float> samples(2 * frames * 800, 0.5f);
    const std::vector<uint8_t> w = wav_bytes(samples.data(), frames * 800);
    assert(w.size() == 44 + size_t(frames) * 800 * 4 && std::string(w.begin(), w.begin() + 4) == "RIFF" && std::string(w.begin() + 8, w.begin() + 16) == "WAVEfmt ");
    assert(w[44] == 0xff && w[45] == 0x3f);   // 0.5 * 32767 = 16383 = 0x3fff, little-endian
    assert(write_file(dir + "/out.wav", w.data(), w.size()));
    { std::ifstream f(dir + "/out.wav", std::ios::binary | std::ios::ate); assert(size_t(f.tellg()) == w.size()); }
    puts("PASS h3 option table, tokenizer buffer growth, output checks and WAV bytes");
}
