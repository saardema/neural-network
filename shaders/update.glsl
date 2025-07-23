#[compute]
#version 430
#include "include/shared.gdshaderinc"
#include "include/buffers.gdshaderinc"

// Weights update pass
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint weight_id = gl_GlobalInvocationID.x;

    // Find which layer this weight belongs to
    uint layer = 0;
    for (uint i = 0; i < layer_count - 1; ++i) {
        if (weight_id < weight_offsets[i] + layer_sizes[i] * layer_sizes[i + 1]) {
            layer = i;
            break;
        }
    }

    // Calculate local indices
    uint weight_start = weight_offsets[layer];
    uint local_weight_id = weight_id - weight_start;
    uint from_neuron = local_weight_id % layer_sizes[layer];
    uint to_neuron = local_weight_id / layer_sizes[layer];

    // Get activation and error values
    uint from_activation_idx = offsets[layer] + from_neuron;
    uint to_error_idx = offsets[layer + 1] + to_neuron;

    // Calculate gradient
    float gradient = activations[from_activation_idx] * errors[to_error_idx];

    // Update weight using gradient descent
    weights[weight_id] += LEARNING_RATE * gradient;
}
