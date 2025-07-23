#[compute]
#version 430
#include "include/shared.gdshaderinc"
#include "include/buffers.gdshaderinc"

// Back propagation pass
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint neuron_id = gl_GlobalInvocationID.x;
    uint layer = 0;

    // Find which layer this neuron belongs to
    uint accumulated_size = 0;
    for (uint i = 0; i < layer_count; ++i) {
        if (neuron_id < accumulated_size + layer_sizes[i]) {
            layer = i;
            break;
        }
        accumulated_size += layer_sizes[i];
    }

    // Skip input layer
    if (layer == 0) return;

    uint local_neuron_id = neuron_id - accumulated_size;

    // Calculate error term
    float error = 0.0;

    if (layer == layer_count - 1) {
        // Output layer - compare with target
        uint target_idx = current_sample * layer_sizes[layer] + local_neuron_id;
        error = target_outputs[target_idx] - activations[neuron_id];
    } else {
        // Hidden layer - backpropagate from next layer
        uint next_layer_size = layer_sizes[layer + 1];
        uint next_weight_start = weight_offsets[layer];

        for (uint k = 0; k < next_layer_size; ++k) {
            uint weight_idx = next_weight_start + k * layer_sizes[layer] + local_neuron_id;
            uint next_error_idx = offsets[layer + 1] + k;
            error += weights[weight_idx] * errors[next_error_idx];
        }
    }

    // Apply derivative of activation function
    errors[neuron_id] = error * sigmoid_derivative(activations[neuron_id]);

    // Update bias using gradient descent
    uint bias_idx = offsets[layer] + local_neuron_id;
    biases[bias_idx] += LEARNING_RATE * errors[neuron_id];
}
