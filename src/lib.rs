#![warn(clippy::pedantic)]

use anyhow::{Error, Result};
use bitvec::prelude::*;
use once_cell::sync::Lazy;
use regex::Regex;
use std::{
    collections::{HashMap, VecDeque},
    fmt::{Debug, Display},
    hash::Hash,
    sync::Mutex,
};

static WORD_SPLIT_REGEX: Lazy<Mutex<Regex>> =
    Lazy::new(|| Mutex::new(Regex::new(r"\w+|\W+").unwrap()));

pub trait Symbol: Clone + Eq + Hash + Display + Default {}

impl<T: Clone + Eq + Hash + Display + Default> Symbol for T {}

#[derive(Debug)]
pub struct HuffmanCodec<T: Symbol> {
    table: HashMap<T, BitBox>,
}

impl<T: Symbol> HuffmanCodec<T> {
    /// Pop either of the two nodes with the smallest weight from the front of
    /// the two queues. If one queue is empty, pop from the other queue. If both
    /// queues are empty, panic.
    fn pop_smallest(a: &mut VecDeque<Node<T>>, b: &mut VecDeque<Node<T>>) -> Node<T> {
        match (a.front(), b.front()) {
            (Some(ai), Some(bi)) => {
                if ai.count <= bi.count {
                    a.pop_front().unwrap()
                } else {
                    b.pop_front().unwrap()
                }
            }
            (Some(_), None) => a.pop_front().unwrap(),
            (None, Some(_)) => b.pop_front().unwrap(),
            (None, None) => panic!("Both queues are empty"),
        }
    }

    fn build_tree(nodes: Vec<Node<T>>) -> Node<T> {
        let mut nodes = nodes;
        nodes.sort_by(|a, b| a.count.cmp(&b.count));
        let mut leaf_q = nodes.into_iter().collect::<VecDeque<_>>();
        let mut internal_q: VecDeque<Node<T>> = VecDeque::new();

        while leaf_q.len() + internal_q.len() > 1 {
            // Combine the nodes into a new internal node
            let internal = Node::new_internal(
                Self::pop_smallest(&mut leaf_q, &mut internal_q),
                Self::pop_smallest(&mut leaf_q, &mut internal_q),
            );

            // Enqueue the new internal node into the rear of the internal queue
            internal_q.push_back(internal);
        }

        // The last node is the root of the Huffman tree
        if let Some(node) = internal_q.pop_front() {
            node
        } else {
            leaf_q.pop_front().expect("No nodes in Huffman tree")
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

    /// Encodes a stream of symbols into a ``BitBox``.
    ///
    /// # Errors
    ///
    /// Will return an error if a symbol is not found in the Huffman table.
    pub fn encode<I: Iterator<Item = T>>(&self, stream: I) -> Result<BitBox> {
        let mut result = BitVec::default();

        for symbol in stream {
            match self.table.get(&symbol) {
                Some(encoded_symbol) => result.extend_from_bitslice(encoded_symbol),
                None => {
                    return Err(Error::msg(format!(
                        "Symbol \"{symbol}\" not found in Huffman table"
                    )))
                }
            }
        }

        Ok(BitBox::from_bitslice(&result))
    }

    /// Decodes a ``BitBox`` into a Vec of symbols.
    ///
    /// # Errors
    ///
    /// Will return an error if the encoded data cannot be fully decoded.
    pub fn decode(&self, encoded: &BitBox) -> Result<Vec<T>> {
        // Build a reverse lookup from BitBox to Symbol
        let lookup: HashMap<&BitBox, &T> = self.table.iter().map(|(k, v)| (v, k)).collect();

        // this lets skip codes that are too short
        let min_code_len = lookup.keys().map(|code| code.len()).min().unwrap_or(1);

        let mut symbols = Vec::new();

        let mut start_idx = 0;
        let mut end_idx = min_code_len;

        while end_idx <= encoded.len() {
            let slice = &encoded[start_idx..end_idx];
            if let Some(symbol) = lookup.get(&BitBox::from_bitslice(slice)) {
                symbols.push((*symbol).clone());
                start_idx = end_idx;
                end_idx += min_code_len - 1; // increment will happen regardless, so subtract 1
            }
            end_idx += 1;
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
        for (symbol, code) in &self.table {
            code_bits += std::mem::size_of_val(code) * 8;
            symbol_size += std::mem::size_of_val(symbol);
        }
        code_bits / 8 + symbol_size
    }

    pub fn symbol_count(&self) -> usize {
        self.table.len()
    }

    pub fn min_max_code_length(&self) -> (usize, usize) {
        let mut min = usize::MAX;
        let mut max = usize::MIN;
        for code in self.table.values() {
            let len = code.len();
            if len < min {
                min = len;
            }
            if len > max {
                max = len;
            }
        }
        (min, max)
    }
}

impl HuffmanCodec<char> {
    pub fn by_char(text: &str) -> Self {
        Self::from_symbols(text.chars())
    }
}

impl HuffmanCodec<String> {
    pub fn split_by_word(text: &str) -> Vec<String> {
        WORD_SPLIT_REGEX
            .lock()
            .unwrap()
            .find_iter(text)
            .map(|matched| matched.as_str().to_string())
            .collect()
    }

    pub fn by_word(text: &str) -> Self {
        Self::from_symbols(Self::split_by_word(text).into_iter())
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

impl<T: Symbol> Drop for Node<T> {
    /// Because Node is a recursive data structure, the default drop
    /// implementation may cause a stack overflow. This implementation uses an
    /// iterative approach.
    fn drop(&mut self) {
        let mut stack = Vec::new();

        match self.kind {
            NodeKind::Leaf { .. } => {}
            NodeKind::Internal {
                ref mut left,
                ref mut right,
            } => {
                stack.push(std::mem::take(left));
                stack.push(std::mem::take(right));
            }
        }

        while let Some(mut node) = stack.pop() {
            match &mut node.kind {
                NodeKind::Leaf { .. } => {}
                NodeKind::Internal {
                    ref mut left,
                    ref mut right,
                } => {
                    stack.push(std::mem::take(left));
                    stack.push(std::mem::take(right));
                }
            }
        }
    }
}
