#extension GL_EXT_scalar_block_layout : require

layout(push_constant, std430) readonly uniform Params {
    uint current_sample;
};

// Network topology (uniform buffer)
layout(std430, binding = 0) readonly uniform NetworkConfig {
    int layer_sizes[MAX_LAYERS];
    int weight_offsets[MAX_LAYERS]; // Starting index for each layer's weights
    int offsets[MAX_LAYERS];        // Starting index for each layer's activations
    int layer_count;
    int neurons_count;
};

layout(std430, binding = 1) restrict buffer WeightBuffer {
    float weights[];          // Flattened weight matrices
};

layout(std430, binding = 2) restrict buffer BiasBuffer {
    float biases[];           // Bias vectors
};

layout(std430, binding = 3) restrict buffer ErrorBuffer {
    float errors[];           // Error terms for backprop
};

layout(std430, binding = 4) restrict readonly buffer TrainingDataBuffer {
    float training_data[];    // Input data
};

layout(std430, binding = 5) restrict readonly buffer TargetOutputBuffer {
    float target_outputs[];   // Expected outputs
};

layout(std430, binding = 0, set = 1) restrict buffer ActivationBuffer {
    float activations[];      // Neuron activations
};

layout(std430, binding = 1, set = 1) restrict writeonly buffer NextActivationBuffer {
    float next_activations[]; // Neuron activations
};