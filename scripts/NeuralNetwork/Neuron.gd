class_name Neuron

const Math = preload('Math.gd')

var activation: float
var pre_activation: float
var bias: float
var delta: float
var weights: PackedFloat32Array
var weight_gradients: PackedFloat32Array
var bias_gradient: float
var incoming: Array[Neuron]
var outgoing: Array[Neuron]
var activation_type := Activations.Type.Sigmoid
var activation_function: Activations.ActivationFunction
var incoming_size: int
var layer: Layer
var learning_rate: float
var neuron_idx: int


func initialize(std_dev: float):
	bias = 0
	delta = 0
	activation = 0
	pre_activation = 0
	bias_gradient = 0
	weights = PackedFloat32Array()
	weight_gradients = PackedFloat32Array()
	for w in incoming_size:
		weights.append(Math.normal_distribution(0, std_dev))
		weight_gradients.append(0)


func compute_activation():
	pre_activation = bias

	for n in incoming_size:
		pre_activation += incoming[n].activation * weights[n]

	activation = activation_function.activate(pre_activation)


func compute_delta():
	delta = 0

	for next_neuron in outgoing:
		delta += next_neuron.delta * next_neuron.weights[neuron_idx]

	delta *= activation_function.differentiate(activation)


func increment_bias_gradient():
	bias_gradient += delta


func average_bias_gradient(batch_size: int):
	bias_gradient /= batch_size


func update_bias():
	bias += delta * learning_rate


func update_bias_from_gradient():
	bias += bias_gradient * learning_rate


func increment_weight_gradients():
	for n in incoming_size:
		weight_gradients[n] += incoming[n].activation * delta


func average_weight_gradients(batch_size: int):
	for n in incoming_size:
		weight_gradients[n] /= batch_size


func update_weights():
	for n in incoming_size:
		weights[n] += incoming[n].activation * delta * learning_rate


func update_weights_from_gradients():
	for n in incoming_size:
		weights[n] += weight_gradients[n] * learning_rate


func reset_gradients():
	bias_gradient = 0
	weight_gradients.fill(0.0)
