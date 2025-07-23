// Network configuration
#define MAX_LAYERS 16
#define MAX_NEURONS_PER_LAYER 256
#define LEARNING_RATE 0.01

// Activation functions
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0 - s);
}

float relu(float x) {
    return max(0.0, x);
}

float relu_derivative(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}