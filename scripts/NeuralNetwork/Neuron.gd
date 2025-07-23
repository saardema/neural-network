class_name Neuron

var activation: float
var pre_activation: float
var bias: float
var error: float
var weights: PackedFloat32Array
var incoming: Array[Neuron]
var outgoing: Array[Neuron]
var activation_type := Activations.Type.Sigmoid
var activation_function: Activations.ActivationFunction


func initialize(std_dev: float, function: Activations.ActivationFunction):
	bias = 0
	error = 0
	activation = 0
	activation_function = function
	for w in weights.size():
		weights[w] = NeuralNetwork.rand_normal(std_dev)


func update_activation():
	pre_activation = bias

	for n in incoming.size():
		pre_activation += incoming[n].activation * weights[n]

	activation = activation_function.activate(pre_activation)


func update_error(neuron_idx: int):
	error = 0

	for neuron in outgoing:
		error += neuron.error * neuron.weights[neuron_idx]

	error *= activation_function.differentiate(activation)


func update_bias(rate: float):
	bias += error * rate


func update_weights(rate: float):
	for n in incoming.size():
		var gradient := incoming[n].activation * error
		weights[n] += gradient * rate
