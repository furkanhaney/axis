use axis::Result;
use axis_imagenet64::prepare_npz_shards;
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let output: PathBuf = args
        .next()
        .ok_or("usage: prepare OUTPUT SHARD.npz [SHARD.npz ...]")?
        .into();
    let shards: Vec<PathBuf> = args.map(PathBuf::from).collect();
    let count = prepare_npz_shards(&shards, &output)?;
    println!("prepared {count} labeled images at {}", output.display());
    Ok(())
}
