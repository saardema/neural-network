const Math = preload('Math.gd')

var activation: float
var pre_activation: float
var bias: float
var delta: float
var weights: PackedFloat32Array
var weight_gradients: PackedFloat32Array
var weight_momentums: PackedFloat32Array
var bias_gradient: float
var prev_bias_gradient: float
var incoming: Array[NeuralNetwork.Neuron]
var outgoing: Array[NeuralNetwork.Neuron]
var activation_type := Activations.Type.Sigmoid
var activation_function: Activations.ActivationFunction
var incoming_size: int
var outgoing_size: int
var layer: NeuralNetwork.Layer
var learning_rate: float
var neuron_idx: int
var gradient_counter: int = 0

func initialize(std_dev: float):
	bias = 0
	delta = 0
	activation = 0
	pre_activation = 0
	bias_gradient = 0
	weights.resize(incoming_size)
	weight_gradients.resize(incoming_size)
	weight_momentums.resize(incoming_size)
	weight_gradients.fill(0.0)
	weight_momentums.fill(0.0)
	for i in incoming_size:
		weights[i] = Math.normal_distribution(0, std_dev)


func compute_activation():
	pre_activation = bias

	for n in incoming_size:
		pre_activation += incoming[n].activation * weights[n]

	activation = activation_function.activate(pre_activation)


func update_bias_from_gradient():
	bias += bias_gradient * learning_rate

func average_bias_gradient(batch_size: int):
	bias_gradient /= batch_size


func average_weight_gradients(batch_size: int):
	for n in incoming_size:
		weight_gradients[n] /= batch_size


func update_weights_from_gradients():
	for n in incoming_size:
		weights[n] += weight_gradients[n] * learning_rate


func reset_gradients():
	bias_gradient = 0
	for w in incoming_size:
		weight_gradients[w] = 0.0
