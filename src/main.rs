#![warn(clippy::pedantic)]

use anyhow::Result;
use huffman_text::HuffmanCodec;
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let path = Path::new("texts/the-complete-works-of-william-shakespeare.txt");
    println!("File: {}", path.display());

    let text = std::fs::read_to_string(path)?;
    let text_bytes = text.len();
    println!("Text Bytes: {text_bytes} B");

    let start_table = Instant::now();
    let table = HuffmanCodec::by_word(&text);
    let table_bytes = table.size();
    println!("Table Build Time: {:?}", start_table.elapsed());
    println!("Table Bytes: {table_bytes} B");

    println!("Symbol Count: {}", table.symbol_count());
    println!("Min/Max Code Length Bits: {:?}", table.min_max_code_length());

    let start_encode = Instant::now();
    let words = HuffmanCodec::split_by_word(&text);
    let encoded = table.encode(words.into_iter())?;
    println!("Encode Time: {:?}", start_encode.elapsed());
    let encoded_bytes = encoded.len() / 8;
    println!("Encoded Size: {encoded_bytes} B");

    let compression_size = table_bytes + encoded_bytes;
    let compression_ratio = text_bytes as f64 / compression_size as f64;
    println!("Compression Ratio: {compression_ratio:.2}");

    let start_decode = Instant::now();
    let decoded_symbols = table.decode(&encoded)?;
    let decoded: String = decoded_symbols.into_iter().collect();
    println!("Decode Time: {:?}", start_decode.elapsed());

    // check equality
    let equal = text.eq(&decoded);
    println!("Decode matches original?: {equal}");

    Ok(())
}
