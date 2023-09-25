# Huffman Compressor

A Huffman compressor/decompressor. A toy project to learn about all sorts a stuff:

- Huffman comp/decomp, obviously
- Recursive structs (and dropping them!)
- Creating traits
- And, I dunno... other stuff too!

See `examples/roundtrip.rs` for a round-trip compression-decompression with stats, or run it with `cargo run --release --example roundtrip`:

![demo](docs/demo.png)

## Process

1. Create a `HuffmanCodec` from an iterator of symbols, which will:
  
   1. Tally the symbols as they are encountered and turn unique symbols into nodes.
   2. Build a Huffman tree from those (leaf) nodes.
   3. Build a table that maps from symbol to Huffman code.

   For simplicity, the iterator should be of the text that will be compressed,
   which ensures optimal frequency determination, and that all symbols will be
   known.

   Some predefined symbol definitions are provided:
     - `char`: split by character
     - `String`: split by word/whitespace (`r"\w+|\W+"`)

2. Encode/Decode

   - Encoding: Turns an iterator of symbols into a `bitvec::BitBox`
   - Decoding: Turns a `bitvec::BitBox` into a vec of symbols

   Unknown symbols or codes will return an error.
