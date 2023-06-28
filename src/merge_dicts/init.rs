use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    // The size of each dict
    #[arg(short, long, default_value_t = 1000000)]
    pub size: i32
}