# webgpu-poc

This repo aims to provide meaningful benchmarks for WebGPU performance in Rust.

## Benchmarks

- [x] Compute Profile
- [ ] Buffer Copy Compute

## Running the benchmarks

Run `cargo run` to run the benchmarks.

## Performance Summary

Currently, the only benchmark is for profiling the entire WebGPU pipeline.

### Run 1M multiplications on MacBook Pro M1 Pro (single thread)

#### Debug

<img src="./assets/compute_profile_1m_multiplications_debug.png" alt="Performance chart showing compute profile benchmark results for 1M multiplications in debug mode" width="500"/>

#### Release

<img src="./assets/compute_profile_1m_multiplications_release.png" alt="Performance chart showing compute profile benchmark results for 1M multiplications in release mode" width="500"/>
