// Import necessary libraries
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rand::Rng;
use std::thread::{self, JoinHandle};
use std::time::Instant;
use clap::Parser;
use super::init::Args;


pub fn run() {
    let mut handles: Vec<JoinHandle<HashMap<String, i32>>> = Vec::new();
    let args = Args::parse();
    let size: i32 = args.size;
    dbg!(&args);
    for _i in 0..28 {
        let handle = thread::spawn(move || {
            let result: HashMap<String, i32> = gen_dicts(size);
            result
        });
        handles.push(handle);
    }
    let mut dicts: Vec<HashMap<String, i32>> = Vec::new();
    // Waiting for the two threads to finish
    let now = Instant::now();
    for handle in handles {
        let dict: HashMap<String, i32> = handle.join().unwrap();
        dicts.push(dict);
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    let now = Instant::now();
    let merged_dict = merge_dicts(&dicts[0], &dicts[1]);
    println!("The length of the merged dict is: {}", merged_dict.len());
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}

fn _process_dict(dict: &Arc<Mutex<HashMap<String, i32>>>) -> HashMap<String, i32> {
    let dict_lock = dict.lock().unwrap();
    for (key, value) in dict_lock.iter() {
        println!("{} => {}", key, value);
    }
    dict_lock.clone()
}

// Function to process a dictionary (HashMap) in Rust
fn merge_dicts(dict1: &HashMap<String, i32>, dict2: &HashMap<String, i32>) -> HashMap<String, i32> {
    println!("Merging HashMaps");
    let mut merged_dict: HashMap<String, i32> = HashMap::new();
    merged_dict.extend(dict1.clone());
    merged_dict.extend(dict2.clone());
    merged_dict
}

fn gen_dicts(_size: i32) -> HashMap<String, i32> {
    let mut dict: HashMap<String, i32> = HashMap::new();
    for i in 0.._size {
        let key: String = format!("{}", rand::thread_rng().gen_range(0..=i32::MAX));
        let value = i;
        dict.insert(key, value);
    }
    println!("The length of the generated dict is: {}", dict.len());
    dict
}
