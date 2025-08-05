class_name NeuralNetwork

signal updated
signal propagated_forwards
signal trained_frame

const Math = preload('Math.gd')

var layers: Array[Layer]
var input_layer: Layer
var output_layer: Layer
var layer_count: int
var loss: float = 0
var train_async: bool = true
var data: TrainingData
var config: NeuralNetworkConfig
var total_iterations: int
var layout: Layout
var activation_functions: Activations
var loss_function: Math.Mean
var mean_gradient: Array[float]
var shuffle_training: bool = true

#region Initialization

func _init(network_layout: Layout, network_config: NeuralNetworkConfig):
	activation_functions = Activations.new()
	layout = network_layout
	config = network_config
	layer_count = layout.hidden.size() + 1
	input_layer = Layer.new()
	input_layer.size = layout.inputs
	mean_gradient.resize(layout.outputs)
	loss_function = Math.Mean.new(Math.Mean.Type.RootMeanSquare)

	for i in input_layer.size:
		input_layer.neurons.append(Neuron.new())

	# Setup layers and populate neurons
	for l in layer_count:
		var layer := Layer.new()
		layer.is_output = l == layer_count - 1
		layer.size = layout.outputs if layer.is_output else layout.hidden[l]
		layers.append(layer)

		for n in layer.size:
			layer.neurons.append(Neuron.new())

	output_layer = layers[-1]

	# Initialize neurons and setup linked list relationships
	input_layer.next = layers[0]
	for i in input_layer.size:
		input_layer.neurons[i].outgoing = layers[0].neurons

	for l in layer_count:
		var layer := layers[l]
		if l > 0: layer.previous = layers[l - 1]
		if l < layer_count - 1: layer.next = layers[l + 1]

		for n in layer.size:
			var neuron := layer.neurons[n]
			neuron.neuron_idx = n
			neuron.incoming = layer.previous.neurons if layer.previous else input_layer.neurons
			neuron.incoming_size = neuron.incoming.size()
			neuron.layer = layer

			if layer.is_output:
				neuron.activation_type = layout.output_activation
			else:
				neuron.activation_type = layout.hidden_activation
			neuron.activation_function = activation_functions.get_function(neuron.activation_type)

			if layer.next: neuron.outgoing = layer.next.neurons
			neuron.initialize(config.wt_std_dev)

func reset():
	data.current_sample = 0
	loss = 1
	initialize_weights()


func initialize_weights():
	for layer in layers:
		for neuron in layer.neurons:
			neuron.initialize(config.wt_std_dev)
	updated.emit()

#endregion


#region Training

func train(iterations := config.iterations):
	loss = 0

	if layers[0].neurons[0].learning_rate != config.learn_rate:
		set_learning_rate(config.learn_rate)

	if config.batch_size > 1: train_mini_batch(iterations)
	else: train_stochastic(iterations)

	loss = loss_function.calculate()
	updated.emit()
	trained_frame.emit()


func set_learning_rate(rate: float):
	for layer in layers:
		for neuron in layer.neurons:
			neuron.learning_rate = rate


func train_mini_batch(iterations: int):
	for i in max(1, iterations / config.batch_size):
		for b in config.batch_size:
			mini_batch_pass()
			update_loss()
		apply_parameters_from_gradients()


func mini_batch_pass():
	propagate_forwards()
	compute_output_error()
	compute_deltas()
	increment_gradients()
	data.next_sample(shuffle_training)


func train_stochastic(iterations: int):
	for i in iterations:
		data.next_sample(shuffle_training)
		propagate_forwards()
		propagate_backwards()
		update_loss()


func propagate_forwards_raw():
	var src := PackedFloat32Array()
	var tgt := PackedFloat32Array()
	var tmp: PackedFloat32Array
	src.resize(layout.max_layer_size)
	tgt.resize(layout.max_layer_size)
	var layer_left: Layer = input_layer
	var hidden_activation = activation_functions.get_function(layout.hidden_activation)
	var output_activation = activation_functions.get_function(layout.output_activation)

	for i in input_layer.size:
		src[i] = data.training_data[data.current_sample * input_layer.size + i]
		input_layer.neurons[i].activation = src[i]

	for layer_right in layers:
		for b in layer_right.size:
			var neuron := layer_right.neurons[b]
			tgt[b] = neuron.bias
			for a in layer_left.size:
				tgt[b] += src[a] * neuron.weights[a]
			if layer_right.is_output:
				neuron.activation = output_activation.activate(tgt[b])
			else:
				neuron.activation = hidden_activation.activate(tgt[b])
				tgt[b] = neuron.activation

		tmp = src
		src = tgt
		tgt = tmp
		layer_left = layer_right


func propagate_forwards():
	for i in input_layer.size:
		input_layer.neurons[i].activation = data.get_input(i)

	for layer in layers:
		for neuron in layer.neurons:
			neuron.compute_activation()

	propagated_forwards.emit()


func propagate_backwards():
	compute_output_error()
	compute_deltas()
	apply_parameters()

func compute_output_error():
	for i in output_layer.size:
		var neuron := output_layer.neurons[i]
		neuron.delta = data.get_target(i) - neuron.activation
		neuron.delta *= neuron.activation_function.differentiate(neuron.activation)


func compute_deltas():
	for l in layer_count - 1:
		for neuron in layers[l].neurons:
			neuron.compute_delta()


func increment_gradients():
	for layer in layers:
		for neuron in layer.neurons:
			neuron.increment_bias_gradient()
			neuron.increment_weight_gradients()


func apply_parameters():
	for layer in layers:
		for neuron in layer.neurons:
			neuron.update_weights()
			neuron.update_bias()


func apply_parameters_from_gradients():
	for layer in layers:
		for neuron in layer.neurons:
			neuron.average_bias_gradient(config.batch_size)
			neuron.average_weight_gradients(config.batch_size)
			neuron.update_weights_from_gradients()
			neuron.update_bias_from_gradient()
			neuron.reset_gradients()


#endregion


#region Helpers

func update_loss():
	for i in output_layer.size:
		loss_function.append(output_layer.neurons[i].delta)

#endregion


#region Sub Classes

class Layout:
	var inputs: int:
		set(v):
			inputs = v
			_update_max_layer_size()

	var hidden: Array[int]:
		set(v):
			hidden = v
			_update_max_layer_size()

	var outputs: int:
		set(v):
			outputs = v
			_update_max_layer_size()

	var hidden_activation: Activations.Type
	var output_activation: Activations.Type
	var max_layer_size: int

	func _init(
		ins: int = 1,
		hidden_layout: Array[int] = [],
		outs: int = 1,
		hid_activation := Activations.Type.Sigmoid,
		out_activation := Activations.Type.Sigmoid):
		hidden_activation = hid_activation
		inputs = ins
		hidden = hidden_layout
		outputs = outs
		output_activation = out_activation

	func _update_max_layer_size():
		max_layer_size = max(inputs, (hidden + [1]).max(), outputs)


#endregion
