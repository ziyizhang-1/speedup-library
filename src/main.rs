// Import necessary libraries
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rand::Rng;
use std::thread::{self, JoinHandle};
use std::time::Instant;

fn main() {
    let mut handles: Vec<JoinHandle<HashMap<String, i32>>> = Vec::new();
    for _i in 0..12 {
        let handle = thread::spawn(move || {
            let result: HashMap<String, i32> = gen_dicts(10000000);
            result.clone()
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
    merge_dicts(&dicts[0], &dicts[1]);
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
fn merge_dicts(dict1: &HashMap<String, i32>, dict2: &HashMap<String, i32>) {
    println!("Merging HashMaps");
    let mut merged_dict: HashMap<String, i32> = HashMap::new();
    merged_dict.extend(dict1.clone());
    merged_dict.extend(dict2.clone());
    println!("The length of the merged dict is: {}", merged_dict.len());
}

fn gen_dicts(_size: i32) -> HashMap<String, i32> {
    let mut dict: HashMap<String, i32> = HashMap::new();
    for i in 0.._size {
        let key: String = format!("{}", rand::thread_rng().gen_range(0..=i32::MAX));
        let value = i;
        dict.insert(key, value);
    }
    println!("The length of the generated dict is: {}", dict.len());
    dict.clone()
}