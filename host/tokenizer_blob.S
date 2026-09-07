// assets/tokenizer.json compiled into libh3pipe (h3tok_create(NULL)); assembled from the repository root by scripts/build_host.sh.
    .section .rodata
    .balign 16
    .globl h3tok_embedded_json
    .globl h3tok_embedded_json_end
h3tok_embedded_json:
    .incbin "assets/tokenizer.json"
h3tok_embedded_json_end:
    .byte 0
    .section .note.GNU-stack,"",@progbits
