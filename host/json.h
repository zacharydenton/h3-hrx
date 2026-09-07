// A minimal JSON reader shared by the tokenizer (tokenizer.json) and the safetensors loader (the file header):
// objects keep their fields in source order, numbers are doubles, strings are decoded (\uXXXX with surrogate pairs).
#ifndef H3_JSON_H
#define H3_JSON_H
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace h3json {

// --- a minimal JSON value ----------------------------------------------------------------------
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double number = 0; std::string str; std::vector<Json> items; std::vector<std::pair<std::string, Json>> fields;
    const Json *get(const std::string &k) const { for (auto &f : fields) if (f.first == k) return &f.second; return nullptr; }
};

struct Parser {
    const std::string &s; size_t i = 0; const char *what;   // what: the document's name for error messages
    explicit Parser(const std::string &src, const char *what_ = "json") : s(src), what(what_) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    [[noreturn]] void fail(const char *m) { throw std::runtime_error(std::string(what) + ": " + m + " at byte " + std::to_string(i)); }
    static void put_utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
        else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    }
    std::string string() {
        if (s[i] != '"') fail("expected string");
        ++i; std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') { ++i; if (i >= s.size()) fail("bad escape"); const char c = s[i++];
                switch (c) { case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break; case 'b': out += '\b'; break; case 'f': out += '\f'; break;
                    case 'u': { if (i + 4 > s.size()) fail("bad \\u"); uint32_t cp = uint32_t(strtoul(s.substr(i, 4).c_str(), nullptr, 16)); i += 4;
                        if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') { const uint32_t lo = uint32_t(strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16)); if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; } }
                        put_utf8(out, cp); break; }
                    default: out += c; } }
            else out += s[i++];
        }
        if (i >= s.size()) fail("unterminated string");
        ++i; return out;
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

inline Json parse(const std::string &text, const char *what) { Parser p(text, what); Json v = p.value(); p.ws(); if (p.i != text.size()) p.fail("trailing characters"); return v; }

}  // namespace h3json
#endif
