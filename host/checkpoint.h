// A read-only view of a safetensors file (ComfyUI's checkpoints as they are): the JSON header (an 8-byte little-endian
// length, then name -> {dtype, shape, data_offsets}) parsed once, the tensor bytes memory-mapped behind it. Nothing is read
// at open beyond the header; pages fault in as tensors are used. Dtype strings are the file's ("I8", "F16", "BF16", "F32").
#ifndef H3_CHECKPOINT_H
#define H3_CHECKPOINT_H
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"

struct Checkpoint {
    struct Entry {
        std::string dtype; std::vector<int64_t> shape; size_t offset = 0, bytes = 0;   // offset into the data block
        size_t rows() const { return shape.empty() ? 1 : size_t(shape[0]); }
        size_t row_bytes() const { return rows() ? bytes / rows() : bytes; }
        size_t elements() const { size_t n = 1; for (int64_t d : shape) n *= size_t(d); return n; }
        std::string describe() const { std::string s = dtype + " ["; for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]); return s + "]"; }
    };
    std::string path; std::map<std::string, Entry> entries;

    explicit Checkpoint(const std::string &file) : path(file) {
        fd_ = open(file.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("cannot open " + file);
        struct stat st;
        if (fstat(fd_, &st) != 0 || st.st_size < 8) { close(fd_); fd_ = -1; throw std::runtime_error("not a safetensors file (shorter than its header length): " + file); }
        size_ = size_t(st.st_size);
        map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) { map_ = nullptr; close(fd_); fd_ = -1; throw std::runtime_error("cannot map " + file); }
        try {
            uint64_t header = 0; memcpy(&header, map_, 8);
            if (header > size_ - 8 || header > (uint64_t(1) << 31)) throw std::runtime_error("corrupt safetensors header length in " + file);
            data_ = static_cast<const char *>(map_) + 8 + header; data_bytes_ = size_ - 8 - size_t(header);
            const h3json::Json root = h3json::parse(std::string(static_cast<const char *>(map_) + 8, size_t(header)), file.c_str());
            if (root.kind != h3json::Json::Object) throw std::runtime_error("safetensors header is not an object: " + file);
            for (const auto &field : root.fields) {
                if (field.first == "__metadata__") continue;
                const h3json::Json &v = field.second;
                const h3json::Json *dtype = v.get("dtype"), *shape = v.get("shape"), *offsets = v.get("data_offsets");
                if (!dtype || dtype->kind != h3json::Json::String || !shape || shape->kind != h3json::Json::Array || !offsets || offsets->kind != h3json::Json::Array || offsets->items.size() != 2)
                    throw std::runtime_error("corrupt tensor entry " + field.first + " in " + file);
                Entry e; e.dtype = dtype->str;
                for (const h3json::Json &d : shape->items) { if (d.number < 0) throw std::runtime_error("negative dimension in " + field.first); e.shape.push_back(int64_t(d.number)); }
                const double b = offsets->items[0].number, en = offsets->items[1].number;
                if (b < 0 || en < b || en > double(data_bytes_)) throw std::runtime_error("tensor span past the data in " + field.first + " (" + file + ")");
                e.offset = size_t(b); e.bytes = size_t(en - b);
                if (e.rows() && e.bytes % e.rows()) throw std::runtime_error("tensor bytes not a multiple of its rows: " + field.first);
                entries.emplace(field.first, std::move(e));
            }
        } catch (...) { munmap(map_, size_); map_ = nullptr; close(fd_); fd_ = -1; throw; }
    }
    ~Checkpoint() { if (map_) munmap(map_, size_); if (fd_ >= 0) close(fd_); }
    Checkpoint(const Checkpoint &) = delete;
    Checkpoint &operator=(const Checkpoint &) = delete;

    bool has(const std::string &name) const { return entries.count(name) != 0; }
    const Entry &at(const std::string &name) const {
        auto it = entries.find(name);
        if (it == entries.end()) throw std::runtime_error("missing tensor " + name + " in " + path);
        return it->second;
    }
    // The tensor with the dtype and shape the loader relies on (a -1 dimension matches anything).
    const Entry &at(const std::string &name, const char *dtype, std::initializer_list<int64_t> shape) const {
        const Entry &e = at(name);
        bool ok = e.dtype == dtype && e.shape.size() == shape.size();
        size_t i = 0; for (int64_t d : shape) { if (ok && d >= 0 && e.shape[i] != d) ok = false; ++i; }
        if (!ok) {
            Entry want; want.dtype = dtype; for (int64_t d : shape) want.shape.push_back(d);
            throw std::runtime_error(name + " is " + e.describe() + ", expected " + want.describe() + " in " + path);
        }
        return e;
    }
    const char *data(const Entry &e) const { return data_ + e.offset; }
    const char *data(const std::string &name) const { return data(at(name)); }
    // Hint the kernel to read a range's pages ahead of a sequential pass over them.
    void will_need(const char *p, size_t bytes) const {
        if (!bytes) return;
        const uintptr_t page = 4096, begin = reinterpret_cast<uintptr_t>(p) & ~(page - 1), end = reinterpret_cast<uintptr_t>(p) + bytes;
        madvise(reinterpret_cast<void *>(begin), size_t(end - begin), MADV_WILLNEED);
    }
    void will_need(const Entry &e) const { will_need(data(e), e.bytes); }

private:
    int fd_ = -1; void *map_ = nullptr; size_t size_ = 0, data_bytes_ = 0; const char *data_ = nullptr;
};
#endif
