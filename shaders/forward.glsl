#[compute]
#version 430
#include "include/shared.gdshaderinc"
#include "include/buffers.gdshaderinc"

// Forward pass
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint neuron_id = gl_GlobalInvocationID.x;
    uint layer = layer_count - 1;

    if (neuron_id >= neurons_count) return;

    // Find which layer this neuron belongs to
    for (uint i = 0; i < layer_count - 1; ++i) {
        if (neuron_id >= offsets[i] && neuron_id < offsets[i+1]) {
            layer = i;
        }
    }

    if (layer == 0) {
        // uint sample_start = current_sample * layer_sizes[0];
        // next_activations[neuron_id] = training_data[sample_start + neuron_id];

        return;
    }

    uint local_neuron_id = neuron_id - offsets[layer];
    uint prev_layer_size = layer_sizes[layer - 1];

    // Calculate weighted sum
    float sum = 0.0;
    uint weight_start = weight_offsets[layer - 1];

    for (uint j = 0; j < prev_layer_size; ++j) {
        uint weight_idx = weight_start + local_neuron_id * prev_layer_size + j;
        uint prev_activation_idx = offsets[layer - 1] + j;
        sum += weights[weight_idx] * activations[prev_activation_idx];
    }

    // Add bias
    uint bias_idx = offsets[layer] - layer_sizes[0] + local_neuron_id;
    sum += biases[bias_idx];

    // Apply activation function
    activations[neuron_id] = sigmoid(sum);
}
