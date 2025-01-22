use wgpu::*;

mod benchmarks;
use benchmarks::{compute_profile::ComputeProfileBenchmark, WebGPUBenchmark};

async fn run() {
    // Create instance
    let instance = Instance::default();

    // Request adapter
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .unwrap();

    // Request device with timestamp query features
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::TIMESTAMP_QUERY
                    | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .unwrap();

    // Run compute profile benchmark with 128MB buffer
    let benchmark = ComputeProfileBenchmark::new(128 * 1024 * 1024);
    println!("\nRunning {}", benchmark.name());
    benchmark.run(&device, &queue, 0).await;
}

fn main() {
    pollster::block_on(run());
}
