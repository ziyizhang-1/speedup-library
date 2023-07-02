use clap::{Parser, ArgAction};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    // The size of each dict
    #[arg(short='s', long, default_value_t=1000000)]
    pub size: usize,
    #[arg(short='q', long, action=ArgAction::SetFalse)]
    pub sequential_only: bool
}