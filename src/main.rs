use anyhow::{Error, Result};
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressIterator};
use once_cell::sync::Lazy;
use regex::Regex;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap},
    fmt::{Debug, Display},
    hash::Hash,
    path::Path,
    sync::Mutex,
};

static WORD_SPLIT_REGEX: Lazy<Mutex<Regex>> =
    Lazy::new(|| Mutex::new(Regex::new(r"\w+|\W+").unwrap()));

// pub trait ByteSizeable {
//     fn len(&self) -> usize;
// }

pub trait Symbol: Clone + Eq + Hash + Display + Debug + Default {}

impl<T: Clone + Eq + Hash + Display + Debug + Default> Symbol for T {}

// impl ByteSizeable for char {
//     fn len(&self) -> usize {
//         1
//     }
// }

// impl ByteSizeable for String {
//     fn len(&self) -> usize {
//         self.len()
//     }
// }

#[derive(Debug)]
pub struct HuffmanCodec<T: Symbol> {
    table: HashMap<T, BitBox>,
}

impl<T: Symbol> HuffmanCodec<T> {
    pub fn build_tree(nodes: Vec<Node<T>>) -> Node<T> {
        let mut leaf_q: BinaryHeap<Reverse<Node<T>>> = nodes.into_iter().map(Reverse).collect();
        let mut internal_q: BinaryHeap<Reverse<Node<T>>> = BinaryHeap::new();

        while leaf_q.len() + internal_q.len() > 1 {
            // Dequeue the two nodes with the lowest weights
            let mut smallest = Vec::with_capacity(2);
            for _ in 0..2 {
                let leaf_top = leaf_q.peek();
                let internal_top = internal_q.peek();

                smallest.push(match (leaf_top, internal_top) {
                    (Some(leaf_count), Some(internal_count)) => {
                        if leaf_count <= internal_count {
                            leaf_q.pop().unwrap().0
                        } else {
                            internal_q.pop().unwrap().0
                        }
                    }
                    (Some(_), None) => leaf_q.pop().unwrap().0,
                    (None, Some(_)) => internal_q.pop().unwrap().0,
                    (None, None) => panic!("Both queues are empty"),
                })
            }

            // Combine the nodes into a new internal node
            let internal = Node::new_internal(smallest.pop().unwrap(), smallest.pop().unwrap());

            // Enqueue the new internal node into the rear of the internal queue
            internal_q.push(std::cmp::Reverse(internal));
        }

        // The last node is the root of the Huffman tree
        if let Some(Reverse(node)) = internal_q.pop() {
            node
        } else {
            leaf_q.pop().unwrap().0
        }
    }

    fn build_table(node: &Node<T>) -> HashMap<T, BitBox> {
        let mut stack = Vec::new();
        let mut table = HashMap::new();

        stack.push((node, BitVec::new()));

        while let Some((node, path)) = stack.pop() {
            match &node.kind {
                NodeKind::Leaf { symbol } => {
                    table.insert(symbol.clone(), BitBox::from_bitslice(&path));
                }
                NodeKind::Internal { left, right } => {
                    let mut left_path = path.clone();
                    left_path.push(false);
                    stack.push((left, left_path));

                    let mut right_path = path.clone();
                    right_path.push(true);
                    stack.push((right, right_path));
                }
            }
        }

        for (symbol, code) in table.iter() {
            let code_disp = code
                .iter()
                .map(|bit| if *bit { '1' } else { '0' })
                .collect::<String>();
            println!("{}: {}", symbol, code_disp);
        }

        table
    }

    pub fn from_symbols<I: Iterator<Item = T>>(symbols: I) -> HuffmanCodec<T> {
        let leaves: Vec<_> = symbols
            .fold(HashMap::new(), |mut map, symbol| {
                *map.entry(symbol).or_insert(0) += 1;
                map
            })
            .into_iter()
            .map(|(symbol, count)| Node::new_leaf(symbol, count))
            .collect();
        let root = Self::build_tree(leaves);
        let table = Self::build_table(&root);
        HuffmanCodec { table }
    }

    pub fn encode<I: Iterator<Item = T>>(&self, stream: I) -> Result<BitBox> {
        let mut result = BitVec::default();

        for symbol in ProgressBar::new(stream.size_hint().0 as u64).wrap_iter(stream) {
            match self.table.get(&symbol) {
                Some(encoded_symbol) => result.extend_from_bitslice(&encoded_symbol),
                None => {
                    return Err(Error::msg(format!(
                        "Symbol \"{}\" not found in Huffman table",
                        symbol
                    )))
                }
            }
        }

        Ok(BitBox::from_bitslice(&result))
    }

    pub fn decode(&self, encoded: &BitBox) -> Result<Vec<T>> {
        // Build a reverse lookup from BitBox to Symbol
        let lookup: HashMap<&BitBox, &T> = self.table.iter().map(|(k, v)| (v, k)).collect();

        // this lets skip codes that are too short
        let min_code_len = lookup.keys().map(|code| code.len()).min().unwrap();
        println!("Min Code Len: {}", min_code_len);
    
        let mut symbols = Vec::new();
    
        let mut start_idx = 0;
        let mut end_idx = min_code_len;

        let progress_bar = ProgressBar::new(encoded.len() as u64);
    
        while end_idx <= encoded.len() {
            let slice = &encoded[start_idx..end_idx];
            if let Some(symbol) = lookup.get(&BitBox::from_bitslice(slice)) {
                symbols.push((*symbol).clone());
                start_idx = end_idx;
                end_idx += min_code_len - 1;  // increment will happen regardless, so subtract 1
            }
            end_idx += 1;
            progress_bar.inc(1);
        }
    
        // Ensure all bits are decoded
        if start_idx != encoded.len() {
            return Err(Error::msg(
                "Invalid encoded data. It cannot be fully decoded.",
            ));
        }
    
        Ok(symbols)
    }

    pub fn size(&self) -> usize {
        let mut code_bits = 0;
        let mut symbol_size = 0;
        for (symbol, code) in self.table.iter() {
            code_bits += std::mem::size_of_val(code) * 8;
            // code_bits += code.len();
            symbol_size += std::mem::size_of_val(symbol);
        }
        code_bits / 8 + symbol_size
    }
}

impl HuffmanCodec<char> {
    pub fn by_char(text: &str) -> Self {
        Self::from_symbols(text.chars())
    }
}

impl HuffmanCodec<String> {
    pub fn by_word(text: &str) -> Self {
        let leaves = WORD_SPLIT_REGEX
            .lock()
            .unwrap()
            .find_iter(text)
            .map(|matched| matched.as_str().to_string())
            .collect::<Vec<String>>();

        Self::from_symbols(leaves.into_iter())
    }
}

#[derive(Debug)]
pub enum NodeKind<T: Symbol> {
    Leaf {
        symbol: T,
    },
    Internal {
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
}

impl<T: Symbol> Default for NodeKind<T> {
    fn default() -> Self {
        NodeKind::Leaf {
            symbol: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct Node<T: Symbol> {
    pub count: u32,
    pub kind: NodeKind<T>,
}

impl<T: Symbol> Node<T> {
    pub fn new_leaf(data: T, count: u32) -> Self {
        Node {
            kind: NodeKind::Leaf { symbol: data },
            count,
        }
    }

    pub fn new_internal(left: Node<T>, right: Node<T>) -> Self {
        Node {
            count: left.count + right.count,
            kind: NodeKind::Internal {
                left: left.into(),
                right: right.into(),
            },
        }
    }
}

// impl partial_eq and partial_ord for Node
impl<T: Symbol> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
    }
}

impl<T: Symbol> Eq for Node<T> {}

impl<T: Symbol> PartialOrd for Node<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.count.cmp(&other.count))
    }
}

impl<T: Symbol> Ord for Node<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count)
    }
}

impl<T: Symbol> Default for Node<T> {
    fn default() -> Self {
        Node {
            count: 0,
            kind: NodeKind::Leaf {
                symbol: Default::default(),
            },
        }
    }
}

// impl<T: Symbol> Drop for Node<T> {
//     fn drop(&mut self) {
//         let mut stack = Vec::new();

//         match self.kind {
//             NodeKind::Leaf { .. } => {}
//             NodeKind::Internal {
//                 ref mut left,
//                 ref mut right,
//             } => {
//                 stack.push(std::mem::take(left));
//                 stack.push(std::mem::take(right));
//             }
//         }

//         while let Some(mut node) = stack.pop() {
//             match &mut node.kind {
//                 NodeKind::Leaf { .. } => {}
//                 NodeKind::Internal {
//                     ref mut left,
//                     ref mut right,
//                 } => {
//                     stack.push(std::mem::take(left));
//                     stack.push(std::mem::take(right));
//                 }
//             }
//         }
//     }
// }


fn main() -> Result<()> {
    let path = Path::new("texts/the-odyssey.txt");
    let text = std::fs::read_to_string(&path)?.chars().collect::<String>();
    let text = "oogaa boogaa boogaa boogaa boogaa boogaa boogaa";
    let text = "a b c d e f g";
    let text_bytes = text.len();
    println!("Text Size: {}", text_bytes);

    let table = HuffmanCodec::by_word(&text);
    let table_bytes = table.size();
    println!("Table Size: {}", table_bytes);

    let encoded = table
        .encode(
            WORD_SPLIT_REGEX
                .lock()
                .unwrap()
                .find_iter(&text)
                .map(|matched| matched.as_str().to_string()),
        )
        .unwrap();
    let encoded_bytes = encoded.len() / 8;
    println!("Encoded Size: {}", encoded_bytes);

    let compression_size = table_bytes + encoded_bytes;
    let compression_ratio = text_bytes as f64 / compression_size as f64;
    println!("Compression Ratio: {:.2}", compression_ratio);

    let decoded: String = table.decode(&encoded).unwrap().into_iter().collect();

    // check equality
    let equal = text.eq(&decoded);
    println!("Decode matches original?: {}", equal);

    Ok(())
}
