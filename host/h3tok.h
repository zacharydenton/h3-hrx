// Byte-level BPE tokenizer for the Qwen3-VL vocabulary (tokenizer.json), so a caller of
// libh3pipe needs no Python: UTF-8 text -> token ids, no special tokens, as
// `AutoTokenizer(prompt, add_special_tokens=False)` for the t2va prompt presentation.
//
// The pre-tokenizer is the GPT-2/Qwen2 split regex evaluated by hand on code points with
// ASCII-exact letter/number/space classes and a conservative Unicode approximation
// (every code point >= 0x80 outside the punctuation, symbol and space ranges counts as a
// letter). Added tokens (<|im_start|> and friends) are not matched; prompts do not contain them.
#ifndef H3TOK_H
#define H3TOK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct h3tok h3tok;
// tokenizer_json: the HF tokenizer.json (BPE model with a ByteLevel pre-tokenizer), or NULL for the tokenizer compiled into the
// library (Qwen's, assets/tokenizer.json; H3_TOKENIZER=<file> overrides it).
h3tok *h3tok_create(const char *tokenizer_json, char *error, size_t error_capacity);
void h3tok_destroy(h3tok *t);
// Returns the number of ids the text encodes to and writes up to `capacity` of them; -1 on error.
int h3tok_encode(const h3tok *t, const char *utf8, int32_t *ids, size_t capacity);
// The vocabulary size (the largest id + 1 among the model's tokens).
int h3tok_vocab_size(const h3tok *t);
#ifdef __cplusplus
}
#endif
#endif
