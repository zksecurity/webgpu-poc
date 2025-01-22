use async_trait::async_trait;
use instant::Instant;
use std::{borrow::Cow, collections::HashMap};
use wgpu::*;

use super::WebGPUBenchmark;

pub struct ComputeProfileBenchmark {
    buffer_size: u64,
}

impl ComputeProfileBenchmark {
    pub fn new(buffer_size: u64) -> Self {
        Self { buffer_size }
    }

    fn print_table_header(title: &str) {
        println!("\n{}", title);
        println!("{:-<60}", "");
        println!("{:<40} {:>15}", "Operation", "Time (ms)");
        println!("{:-<60}", "");
    }

    fn print_table_row(name: &str, duration: std::time::Duration) {
        println!("{:<40} {:>15.3}", name, duration.as_secs_f64() * 1000.0);
    }

    fn print_gpu_table_row(name: &str, duration_ns: u64) {
        println!("{:<40} {:>15.3}", name, duration_ns as f64 / 1_000_000.0);
    }

    fn print_table_footer() {
        println!("{:-<60}", "");
    }
}

#[derive(Debug)]
struct BenchmarkResults {
    gpu_compute_duration_ns: u64,
    compute_pass_duration_ns: u64,
    input_copy_duration_ns: u64,
    output_copy_duration_ns: u64,
    buffer_size: u64,
    resource_creation_times: Vec<(String, std::time::Duration)>,
    command_buffer_times: Vec<(String, std::time::Duration)>,
}

impl BenchmarkResults {
    fn print(&self) {
        ComputeProfileBenchmark::print_table_header("GPU Resource Creation Times");
        for (name, duration) in &self.resource_creation_times {
            ComputeProfileBenchmark::print_table_row(name, *duration);
        }

        ComputeProfileBenchmark::print_table_header("Command Buffer Operations");
        for (name, duration) in &self.command_buffer_times {
            ComputeProfileBenchmark::print_table_row(name, *duration);
        }

        ComputeProfileBenchmark::print_table_header("GPU Operation Times");
        ComputeProfileBenchmark::print_gpu_table_row("Compute", self.gpu_compute_duration_ns);
        ComputeProfileBenchmark::print_gpu_table_row(
            &format!("Input buffer copy (size: {} MB)", self.buffer_size >> 20),
            self.input_copy_duration_ns,
        );
        ComputeProfileBenchmark::print_gpu_table_row(
            &format!("Output buffer copy (size: {} MB)", self.buffer_size >> 20),
            self.output_copy_duration_ns,
        );

        let total_cpu_time_ms = self
            .resource_creation_times
            .iter()
            .map(|(_, duration)| duration.as_secs_f64() * 1000.0)
            .sum::<f64>();
        let total_gpu_time_ms = self
            .command_buffer_times
            .iter()
            .find(|(name, _)| name == "Submit command buffer and wait")
            .unwrap()
            .1
            .as_secs_f64()
            * 1000.0;
        let total_bytes = self.buffer_size * 2;
        let throughput_gbps = (total_bytes as f64 / total_gpu_time_ms / 1_000_000.0) * 8.0;

        ComputeProfileBenchmark::print_table_header("Performance Summary");
        let total_time = total_cpu_time_ms + total_gpu_time_ms;
        println!("{:<40} {:>15.2} {:>10.1}%", "Total Time", total_time, 100.0);
        println!(
            "{:<40} {:>15.2} {:>10.1}%",
            "CPU Time",
            total_cpu_time_ms,
            (total_cpu_time_ms / total_time) * 100.0
        );
        println!(
            "{:<40} {:>15.2} {:>10.1}%",
            "GPU Time",
            total_gpu_time_ms,
            (total_gpu_time_ms / total_time) * 100.0
        );
        println!(
            "{:<40} {:>15.2} {:>10.1}%",
            "GPU Compute Time (1M multiplications)",
            self.gpu_compute_duration_ns as f64 / 1_000_000.0,
            (self.gpu_compute_duration_ns as f64 / 1_000_000.0 / total_time) * 100.0
        );
        println!("{:<40} {:>15.2}", "Throughput (Gbps)", throughput_gbps);
        ComputeProfileBenchmark::print_table_footer();
    }
}

#[async_trait]
impl WebGPUBenchmark for ComputeProfileBenchmark {
    fn name(&self) -> &str {
        "WebGPU Compute Profile"
    }

    async fn run(&self, device: &Device, queue: &Queue, _buffer_size: u64) {
        let result1 = self
            .run_shader(device, queue, self.buffer_size, "no_read_no_write_x1")
            .await;

        let result2 = self
            .run_shader(device, queue, self.buffer_size, "no_read_no_write_x2")
            .await;

        // Needed to isolate the GPU compute time by subtracting the time for 2M multiplications from the time for 1M multiplications
        let result = self.post_process_results(result1, result2);
        result.print();
    }
}

impl ComputeProfileBenchmark {
    async fn run_shader(
        &self,
        device: &Device,
        queue: &Queue,
        buffer_size: u64,
        shader_name: &str,
    ) -> BenchmarkResults {
        println!("\n>>> Starting resource creation phase");
        let mut resource_creation_times = Vec::new();
        let mut command_buffer_times = Vec::new();

        println!(">>> Creating timestamp query set");
        let start = Instant::now();
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: None,
            count: 8,
            ty: QueryType::Timestamp,
        });
        resource_creation_times.push(("Create query set".to_string(), start.elapsed()));

        println!(">>> Creating and filling input buffer");
        let start = Instant::now();
        let input_buffer = {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Input Buffer"),
                size: buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            // Fill input buffer with test data
            {
                let mut data = buffer.slice(..).get_mapped_range_mut();
                for (i, chunk) in data.chunks_mut(4).enumerate() {
                    chunk.copy_from_slice(&(i as u32).to_ne_bytes());
                }
            }
            buffer.unmap();
            buffer
        };
        resource_creation_times.push((
            format!(
                "Create and fill input buffer (size: {} MB)",
                buffer_size >> 20
            )
            .to_string(),
            start.elapsed(),
        ));

        // Create staging input buffer
        let start = Instant::now();
        let staging_input = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Input Buffer"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        resource_creation_times.push(("Create staging input buffer".to_string(), start.elapsed()));

        // Create output buffer
        let start = Instant::now();
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Output Buffer"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        resource_creation_times.push(("Create output buffer".to_string(), start.elapsed()));

        // Create staging output buffer
        let start = Instant::now();
        let staging_output = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Output Buffer"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        resource_creation_times.push(("Create staging output buffer".to_string(), start.elapsed()));

        // Create shader module and pipeline
        let start = Instant::now();
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: match shader_name {
                "no_read_no_write_x1" => ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "./shaders/no_read_no_write_x1.wgsl"
                ))),
                "no_read_no_write_x2" => ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "./shaders/no_read_no_write_x2.wgsl"
                ))),
                _ => panic!("Unknown shader: {}", shader_name),
            },
        });
        resource_creation_times.push(("Create shader module".to_string(), start.elapsed()));

        // Create bind group layout
        let start = Instant::now();
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        resource_creation_times.push(("Create bind group layout".to_string(), start.elapsed()));

        // Create pipeline layout
        let start = Instant::now();
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        resource_creation_times.push(("Create pipeline layout".to_string(), start.elapsed()));

        // Create bind group
        let start = Instant::now();
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        resource_creation_times.push(("Create bind group".to_string(), start.elapsed()));

        // Create compute pipeline
        let start = Instant::now();
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::from([("buffer_size".to_string(), buffer_size as f64)]),
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        resource_creation_times.push(("Create compute pipeline".to_string(), start.elapsed()));

        // Create query buffers
        let start = Instant::now();
        let query_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Query Buffer"),
            size: 64, // 8 timestamps * 8 bytes
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Timestamp Buffer"),
            size: 64, // 8 timestamps * 8 bytes
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        resource_creation_times.push(("Create query buffers".to_string(), start.elapsed()));

        command_buffer_times.push(("Create command encoder".to_string(), start.elapsed()));

        println!(">>> Starting command buffer recording");
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        println!(">>> Recording compute pass commands");
        encoder.write_timestamp(&query_set, 2);
        encoder.copy_buffer_to_buffer(&input_buffer, 0, &staging_input, 0, buffer_size);
        encoder.write_timestamp(&query_set, 3);

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.write_timestamp(&query_set, 4);
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_output, 0, buffer_size);
        encoder.write_timestamp(&query_set, 5);

        encoder.resolve_query_set(&query_set, 0..6, &query_buffer, 0);
        encoder.copy_buffer_to_buffer(&query_buffer, 0, &timestamp_buffer, 0, 64);
        command_buffer_times.push(("Record commands".to_string(), start.elapsed()));

        let command_buffer = encoder.finish();

        println!(">>> Submitting command buffer to GPU");
        let start = Instant::now();
        queue.submit(Some(command_buffer));
        device.poll(wgpu::MaintainBase::Wait);
        command_buffer_times.push((
            "Submit command buffer and wait".to_string(),
            start.elapsed(),
        ));

        println!(">>> Reading timestamp data");
        let timestamp_slice = timestamp_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        timestamp_slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::MaintainBase::Wait);

        println!(">>> Finished shader execution");
        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let timestamp_data = timestamp_slice.get_mapped_range();
            let timestamps: [u64; 6] = [
                u64::from_ne_bytes(timestamp_data[0..8].try_into().unwrap()),
                u64::from_ne_bytes(timestamp_data[8..16].try_into().unwrap()),
                u64::from_ne_bytes(timestamp_data[16..24].try_into().unwrap()),
                u64::from_ne_bytes(timestamp_data[24..32].try_into().unwrap()),
                u64::from_ne_bytes(timestamp_data[32..40].try_into().unwrap()),
                u64::from_ne_bytes(timestamp_data[40..48].try_into().unwrap()),
            ];

            let compute_pass_duration_ns = timestamps[1].saturating_sub(timestamps[0]);
            let input_copy_duration_ns = timestamps[3].saturating_sub(timestamps[2]);
            let output_copy_duration_ns = timestamps[5].saturating_sub(timestamps[4]);

            drop(timestamp_data);
            timestamp_buffer.unmap();

            BenchmarkResults {
                gpu_compute_duration_ns: 0, // will be computed in post_process_results
                compute_pass_duration_ns,
                input_copy_duration_ns,
                output_copy_duration_ns,
                buffer_size,
                resource_creation_times,
                command_buffer_times,
            }
        } else {
            println!("!!! Failed to read timestamp data");
            panic!("Failed to read timestamp data");
        }
    }

    fn post_process_results(
        &self,
        x1_result: BenchmarkResults,
        x2_result: BenchmarkResults,
    ) -> BenchmarkResults {
        let x1_gpu_compute_duration_ns = x1_result
            .compute_pass_duration_ns
            .saturating_sub(x1_result.output_copy_duration_ns);
        let x2_gpu_compute_duration_ns = x2_result
            .compute_pass_duration_ns
            .saturating_sub(x2_result.output_copy_duration_ns);
        let gpu_compute_duration_ns =
            x2_gpu_compute_duration_ns.saturating_sub(x1_gpu_compute_duration_ns);
        return BenchmarkResults {
            gpu_compute_duration_ns,
            compute_pass_duration_ns: (x1_result.compute_pass_duration_ns
                + x2_result.compute_pass_duration_ns)
                / 2,
            input_copy_duration_ns: (x1_result.input_copy_duration_ns
                + x2_result.input_copy_duration_ns)
                / 2,
            output_copy_duration_ns: (x1_result.output_copy_duration_ns
                + x2_result.output_copy_duration_ns)
                / 2,
            buffer_size: (x1_result.buffer_size + x2_result.buffer_size) / 2,
            resource_creation_times: x1_result
                .resource_creation_times
                .into_iter()
                .zip(x2_result.resource_creation_times)
                .map(|(a, b)| (a.0, (a.1 + b.1) / 2))
                .collect(),
            command_buffer_times: x1_result
                .command_buffer_times
                .into_iter()
                .zip(x2_result.command_buffer_times)
                .map(|(a, b)| (a.0, (a.1 + b.1) / 2))
                .collect(),
        };
    }
}
