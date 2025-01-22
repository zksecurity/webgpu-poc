use async_trait::async_trait;
use wgpu::*;

pub mod buffer_copy_compute;
pub mod compute_profile;

pub const NUM_ITERATIONS: u32 = 100;
pub const BUFFER_SIZES: &[u64] = &[
    1024,              // 1KB
    1024 * 1024,       // 1MB
    128 * 1024 * 1024, // 128MB
];

// Trait for implementing benchmarks
#[async_trait]
pub trait WebGPUBenchmark {
    fn name(&self) -> &str;
    async fn run(&self, device: &Device, queue: &Queue, buffer_size: u64);
}
