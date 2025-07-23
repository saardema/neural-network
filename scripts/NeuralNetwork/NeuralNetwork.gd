class_name NeuralNetwork

signal network_updated
signal propagated_forwards

var layers: Array[Layer]
var input_layer: Layer
var output_layer: Layer
var layer_count: int
var loss: float = 0
var train_async: bool = false
var data: TrainingData
var config: NeuralNetworkConfig
var total_iterations: int
var layout: Layout
var activation_functions := Activations.new()


func _init(network_layout: Layout, network_config: NeuralNetworkConfig):
	layout = network_layout
	config = network_config
	layer_count = layout.hidden.size() + 1
	input_layer = Layer.new()
	input_layer.size = layout.inputs

	for i in input_layer.size:
		input_layer.neurons.append(Neuron.new())

	# Setup layers and populate neurons
	for l in layer_count:
		var layer := Layer.new()
		layer.is_output = l == layer_count - 1
		layer.size = layout.outputs if layer.is_output else layout.hidden[l]
		layers.append(layer)

		for n in layer.size:
			var neuron := Neuron.new()
			if layer.is_output:
				neuron.activation_type = layout.output_activation
			layer.neurons.append(neuron)

	output_layer = layers[-1]

	# Initialize neurons and setup linked list relationships
	input_layer.next = layers[0]
	for i in input_layer.size:
		input_layer.neurons[i].outgoing = layers[0].neurons

	for l in layer_count:
		var layer := layers[l]
		if l > 0: layer.previous = layers[l - 1]
		if l < layer_count - 1: layer.next = layers[l + 1]

		for neuron in layer.neurons:
			neuron.incoming = layer.previous.neurons if layer.previous else input_layer.neurons
			neuron.weights.resize(neuron.incoming.size())
			neuron.initialize(
				config.wt_std_dev,
				activation_functions.get_function(neuron.activation_type)
			)
			if layer.next: neuron.outgoing = layer.next.neurons

func reset():
	data.current_sample = 0
	loss = 1
	initialize_weights()

func initialize_weights():
	for layer in layers:
		for neuron in layer.neurons:
			neuron.initialize(
				config.wt_std_dev,
				activation_functions.get_function(neuron.activation_type)
			)
	network_updated.emit()


func set_training_data(training_data: TrainingData):
	data = training_data

func train():
	if train_async:
		var task_id := WorkerThreadPool.add_task(_train)
		WorkerThreadPool.wait_for_task_completion(task_id)
	else:
		_train()

	network_updated.emit()


func _train():
	loss = 0

	for i in config.iterations:
		propagate_forwards()
		propagate_backwards()
		data.next_sample()
	loss = abs(loss)


func step_fwd():
	propagate_forwards()
	network_updated.emit()


func step_bwd():
	propagate_backwards()
	network_updated.emit()


func next_sample(shuffle: bool = false):
	data.next_sample(shuffle)
	network_updated.emit()


func propagate_forwards():
	for i in input_layer.size:
		input_layer.neurons[i].activation = data.get_input(i)

	for layer in layers:
		for neuron in layer.neurons: neuron.update_activation()

	calculate_loss()
	propagated_forwards.emit()

func calculate_loss():
	loss = 0
	for i in output_layer.size:
		var error := data.get_target(i) - output_layer.neurons[i].activation
		error **= 2
		loss += error
	loss = 0.5 * loss / output_layer.size


func propagate_backwards():
	for i in output_layer.size:
		var neuron := output_layer.neurons[i]
		neuron.error = data.get_target(i) - neuron.pre_activation
		neuron.error *= neuron.activation_function.differentiate(neuron.activation)

	for l in range(layer_count - 1, -1, -1):
		for n in layers[l].size:
			if l < layer_count - 1:
				layers[l].neurons[n].update_error(n)
			layers[l].neurons[n].update_weights(config.learn_rate)
			layers[l].neurons[n].update_bias(config.learn_rate)


static func normal_distribution(mean: float, sd: float):
	return (PI * sd) * exp(-0.5 * ((randf() - mean) / sd) ** 2)


static func rand_normal(std_dev: float = 0.01):
	var u1 = randf()
	var u2 = randf()

	return sqrt(-2.0 * log(max(u1, 1e-7))) * cos(2.0 * PI * u2) * std_dev


class Layout:
	var inputs: int
	var hidden: Array[int]
	var outputs: int
	var output_activation: Activations.Type

	func _init(
		ins: int = 1,
		hidden_layout: Array[int] = [],
		outs: int = 1,
		activation := Activations.Type.Sigmoid):
		inputs = ins
		hidden = hidden_layout
		outputs = outs
		output_activation = activation
