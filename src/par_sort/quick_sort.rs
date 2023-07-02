use clap::Parser;
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use super::init::Args;
use std::time::Instant;
use rayon::prelude::*;

pub trait Joiner {
    fn is_parallel() -> bool;
    fn join<A, R_A, B, R_B>(oper_a: A, oper_b: B) -> (R_A, R_B)
    where
        A: FnOnce() -> R_A + Send,
        B: FnOnce() -> R_B + Send,
        R_A: Send,
        R_B: Send;
}

pub struct Parallel;
impl Joiner for Parallel {
    #[inline]
    fn is_parallel() -> bool {
        true
    }
    #[inline]
    fn join<A, R_A, B, R_B>(oper_a: A, oper_b: B) -> (R_A, R_B)
    where
        A: FnOnce() -> R_A + Send,
        B: FnOnce() -> R_B + Send,
        R_A: Send,
        R_B: Send,
    {
        rayon::join(oper_a, oper_b)
    }
}

struct Sequential;
impl Joiner for Sequential {
    #[inline]
    fn is_parallel() -> bool {
        false
    }
    #[inline]
    fn join<A, R_A, B, R_B>(oper_a: A, oper_b: B) -> (R_A, R_B)
    where
        A: FnOnce() -> R_A + Send,
        B: FnOnce() -> R_B + Send,
        R_A: Send,
        R_B: Send,
    {
        let a = oper_a();
        let b = oper_b();
        (a, b)
    }
}

fn gen_vec(n: usize) -> Vec<u32> {
    let mut seed = <XorShiftRng as SeedableRng>::Seed::default();
    (0..).zip(seed.as_mut()).for_each(|(i, x)| *x = i);
    let rng = XorShiftRng::from_seed(seed);
    rng.sample_iter(&Standard).take(n).collect()
}

pub fn run() {
    let args = Args::parse();
    let mut base_vec: Vec<u32> = gen_vec(args.size);
    let mut sort_vec: Vec<u32> = base_vec.clone();
    dbg!(&sort_vec[0..10]);
    let now = Instant::now();
    match args.sequential_only {
        true => {
            quick_sort::<Parallel, u32>(&mut sort_vec);
            let elapsed = now.elapsed();
            println!("Hand-crafted QuickSort Elapsed: {:.2?}", elapsed);
            dbg!(&sort_vec[0..10]);
            let now = Instant::now();
            base_vec.par_sort();
            let elapsed = now.elapsed();
            println!("ParSort Elapsed: {:.2?}", elapsed);
            dbg!(&base_vec[0..10]);
        },
        false => {
            quick_sort::<Sequential, u32>(&mut sort_vec);
            let elapsed = now.elapsed();
            println!("Elapsed: {:.2?}", elapsed);
            dbg!(&sort_vec[0..10]);
        }
    }
}

fn quick_sort<J: Joiner, T: PartialOrd + Send>(v: &mut [T]) {
    if v.len() <= 1 {
        return;
    }

    if J::is_parallel() && v.len() <= 5 * 1024 {
        return quick_sort::<Sequential, T>(v);
    }

    let mid = partition(v);
    let (lo, hi) = v.split_at_mut(mid);
    J::join(|| quick_sort::<J, T>(lo), || quick_sort::<J, T>(hi));
}

fn partition<T: PartialOrd + Send>(v: &mut [T]) -> usize {
    let pivot = v.len() - 1;
    let mut i = 0;
    for j in 0..pivot {
        if v[j] <= v[pivot] {
            v.swap(i, j);
            i += 1;
        }
    }
    v.swap(i, pivot);
    i
}