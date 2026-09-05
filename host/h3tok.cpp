// Byte-level BPE (GPT-2 / Qwen2 family) from tokenizer.json: a small JSON reader for the
// vocab and merges, the byte -> unicode table, the split regex evaluated by hand, and the
// rank-driven merge loop. See h3tok.h.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "h3tok.h"

namespace {

// --- a minimal JSON value ----------------------------------------------------------------------
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double number = 0; std::string str; std::vector<Json> items; std::vector<std::pair<std::string, Json>> fields;
    const Json *get(const std::string &k) const { for (auto &f : fields) if (f.first == k) return &f.second; return nullptr; }
};

struct Parser {
    const std::string &s; size_t i = 0;
    explicit Parser(const std::string &src) : s(src) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    [[noreturn]] void fail(const char *m) { throw std::runtime_error(std::string("tokenizer.json: ") + m + " at byte " + std::to_string(i)); }
    static void put_utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
        else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    }
    std::string string() {
        if (s[i] != '"') fail("expected string"); ++i; std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') { ++i; if (i >= s.size()) fail("bad escape"); const char c = s[i++];
                switch (c) { case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break; case 'b': out += '\b'; break; case 'f': out += '\f'; break;
                    case 'u': { if (i + 4 > s.size()) fail("bad \\u"); uint32_t cp = uint32_t(strtoul(s.substr(i, 4).c_str(), nullptr, 16)); i += 4;
                        if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') { const uint32_t lo = uint32_t(strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16)); if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; } }
                        put_utf8(out, cp); break; }
                    default: out += c; } }
            else out += s[i++];
        }
        if (i >= s.size()) fail("unterminated string"); ++i; return out;
    }
    Json value(int depth = 0) {
        ws(); if (i >= s.size()) fail("unexpected end"); Json v;
        if (s[i] == '{') { v.kind = Json::Object; ++i; ws(); if (s[i] == '}') { ++i; return v; }
            while (true) { ws(); std::string k = string(); ws(); if (s[i] != ':') fail("expected :"); ++i; v.fields.push_back({k, value(depth + 1)}); ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == '}') { ++i; return v; } fail("expected , or }"); } }
        if (s[i] == '[') { v.kind = Json::Array; ++i; ws(); if (s[i] == ']') { ++i; return v; }
            while (true) { v.items.push_back(value(depth + 1)); ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == ']') { ++i; return v; } fail("expected , or ]"); } }
        if (s[i] == '"') { v.kind = Json::String; v.str = string(); return v; }
        if (s.compare(i, 4, "true") == 0) { v.kind = Json::Bool; v.number = 1; i += 4; return v; }
        if (s.compare(i, 5, "false") == 0) { v.kind = Json::Bool; i += 5; return v; }
        if (s.compare(i, 4, "null") == 0) { i += 4; return v; }
        { char *end = nullptr; v.number = strtod(s.c_str() + i, &end); if (end == s.c_str() + i) fail("bad token"); i = size_t(end - s.c_str()); v.kind = Json::Number; return v; }
    }
};

// --- code point classes for the split regex -----------------------------------------------------
uint32_t decode_utf8(const std::string &s, size_t &i) {
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { ++i; return c; }
    int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0; uint32_t cp = c & (0x3F >> n);
    ++i; for (int k = 0; k < n && i < s.size(); ++k, ++i) cp = (cp << 6) | ((unsigned char)s[i] & 0x3F);
    return cp;
}
bool is_space(uint32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000; }
bool is_number(uint32_t c) { return (c >= '0' && c <= '9') || (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F) || (c >= 0xFF10 && c <= 0xFF19) || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) || (c >= 0x2150 && c <= 0x218F) || (c >= 0x2460 && c <= 0x249B); }
bool is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (is_space(c) || is_number(c)) return false;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c < 0xC0) return false;                                                    // Latin-1 punctuation and symbols
    if (c == 0xD7 || c == 0xF7) return false;
    if ((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF00 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65)) return false;   // punctuation, symbols, arrows, box drawing
    if (c >= 0x1F000 && c <= 0x1FAFF) return false;                                  // emoji and symbols
    if (c >= 0xE000 && c <= 0xF8FF) return false;                                    // private use
    return true;
}

// the split regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<std::string> pretokenize(const std::string &text) {
    std::vector<uint32_t> cps; std::vector<size_t> offs;
    for (size_t i = 0; i < text.size();) { offs.push_back(i); cps.push_back(decode_utf8(text, i)); }
    offs.push_back(text.size());
    const size_t n = cps.size(); std::vector<std::string> out; size_t p = 0;
    auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    auto is_nl = [](uint32_t c) { return c == '\r' || c == '\n'; };
    while (p < n) {
        size_t q = p; const uint32_t c = cps[p];
        if (c == '\'' && p + 1 < n) {
            const uint32_t a = lower(cps[p + 1]), b = p + 2 < n ? lower(cps[p + 2]) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') q = p + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) q = p + 3;
        }
        if (q == p) {
            if (is_letter(c) || (!is_nl(c) && !is_number(c) && p + 1 < n && is_letter(cps[p + 1]))) {     // [^\r\n\p{L}\p{N}]?\p{L}+
                q = is_letter(c) ? p : p + 1; while (q < n && is_letter(cps[q])) ++q;
            } else if (is_number(c)) q = p + 1;                                                             // \p{N}
            else if (!is_space(c) || (c == ' ' && p + 1 < n && !is_space(cps[p + 1]) && !is_letter(cps[p + 1]) && !is_number(cps[p + 1]))) {   //  ?[^\s\p{L}\p{N}]+[\r\n]*
                q = c == ' ' ? p + 1 : p; while (q < n && !is_space(cps[q]) && !is_letter(cps[q]) && !is_number(cps[q])) ++q; while (q < n && is_nl(cps[q])) ++q;
            } else {                                                                                        // whitespace runs
                size_t r = p; while (r < n && is_space(cps[r])) ++r;
                size_t last_nl = 0; bool any_nl = false; for (size_t k = p; k < r; ++k) if (is_nl(cps[k])) { any_nl = true; last_nl = k; }
                if (any_nl) q = last_nl + 1;                                                                // \s*[\r\n]+
                else if (r < n && r - p > 1) q = r - 1;                                                     // \s+(?!\S): leave one space for the next word
                else q = r;                                                                                 // \s+
            }
        }
        if (q <= p) q = p + 1;
        out.push_back(text.substr(offs[p], offs[q] - offs[p])); p = q;
    }
    return out;
}

struct Tok {
    std::unordered_map<std::string, int> vocab; std::unordered_map<std::string, int> ranks; std::string byte_char[256]; int vocab_size = 0;
    std::unordered_map<std::string, std::vector<int32_t>> cache;
    explicit Tok(const std::string &path) {
        std::ifstream f(path); if (!f) throw std::runtime_error("cannot read " + path);
        std::stringstream ss; ss << f.rdbuf(); const std::string src = ss.str();
        Parser ps(src); Json root = ps.value();
        const Json *model = root.get("model"); if (!model || model->kind != Json::Object) throw std::runtime_error("tokenizer.json: no model");
        const Json *type = model->get("type"); if (!type || type->str != "BPE") throw std::runtime_error("tokenizer.json: model is not BPE");
        const Json *v = model->get("vocab"); if (!v) throw std::runtime_error("tokenizer.json: no vocab");
        for (auto &f2 : v->fields) { vocab[f2.first] = int(f2.second.number); vocab_size = std::max(vocab_size, int(f2.second.number) + 1); }
        const Json *m = model->get("merges"); if (!m) throw std::runtime_error("tokenizer.json: no merges");
        int rank = 0;
        for (auto &it : m->items) {
            if (it.kind == Json::String) ranks[it.str] = rank++;                                  // "a b"
            else if (it.kind == Json::Array && it.items.size() == 2) ranks[it.items[0].str + " " + it.items[1].str] = rank++;
        }
        // GPT-2's byte -> unicode table: printable bytes map to themselves, the rest to 256 + n
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            std::string s; Parser::put_utf8(s, printable ? uint32_t(b) : uint32_t(256 + n++)); byte_char[b] = s;
        }
    }
    void encode_word(const std::string &word, std::vector<int32_t> &out) {
        auto hit = cache.find(word); if (hit != cache.end()) { out.insert(out.end(), hit->second.begin(), hit->second.end()); return; }
        std::vector<std::string> parts; for (unsigned char b : word) parts.push_back(byte_char[b]);
        while (parts.size() > 1) {
            int best = -1; size_t at = 0;
            for (size_t i = 0; i + 1 < parts.size(); ++i) { auto r = ranks.find(parts[i] + " " + parts[i + 1]); if (r != ranks.end() && (best < 0 || r->second < best)) { best = r->second; at = i; } }
            if (best < 0) break;
            parts[at] += parts[at + 1]; parts.erase(parts.begin() + at + 1);
        }
        std::vector<int32_t> ids;
        for (auto &p : parts) { auto it = vocab.find(p); if (it == vocab.end()) throw std::runtime_error("piece not in vocab: " + p); ids.push_back(it->second); }
        if (cache.size() < 65536) cache[word] = ids;
        out.insert(out.end(), ids.begin(), ids.end());
    }
    std::vector<int32_t> encode(const std::string &text) {
        std::vector<int32_t> out;
        for (auto &w : pretokenize(text)) encode_word(w, out);
        return out;
    }
};

}  // namespace

struct h3tok { Tok value; explicit h3tok(const std::string &p) : value(p) {} };

extern "C" h3tok *h3tok_create(const char *path, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    try { if (!path) throw std::invalid_argument("tokenizer_json is required"); return new h3tok(path); }
    catch (const std::exception &e) { if (error && cap) snprintf(error, cap, "%s", e.what()); return nullptr; }
}
extern "C" void h3tok_destroy(h3tok *t) { delete t; }
extern "C" int h3tok_vocab_size(const h3tok *t) { return t ? t->value.vocab_size : 0; }
extern "C" int h3tok_encode(const h3tok *t, const char *utf8, int32_t *ids, size_t capacity) {
    if (!t || !utf8) return -1;
    try {
        std::vector<int32_t> v = const_cast<Tok &>(t->value).encode(utf8);
        for (size_t i = 0; i < v.size() && i < capacity; ++i) ids[i] = v[i];
        return int(v.size());
    } catch (...) { return -1; }
}
