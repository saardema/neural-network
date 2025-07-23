#[compute]
#version 430
#include "include/shared.gdshaderinc"
#include "include/buffers.gdshaderinc"

// Initial input activation pass
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint input_neuron_id = gl_GlobalInvocationID.x;

    if (input_neuron_id >= layer_sizes[0]) return;

    uint sample_start = current_sample * layer_sizes[0];
    activations[input_neuron_id] = training_data[sample_start + input_neuron_id];
    next_activations[input_neuron_id] = activations[input_neuron_id];
}