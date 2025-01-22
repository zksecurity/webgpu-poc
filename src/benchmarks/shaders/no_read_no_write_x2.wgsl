override buffer_size: u32;
@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_data)) {
        return;
    }
    
    // Do 2 million multiplications
    var result: u32 = 1u;
    for (var i: u32 = 0u; i < 1000000u * 2u; i = i + 1u) {
        result = result * 3u;
    }
}