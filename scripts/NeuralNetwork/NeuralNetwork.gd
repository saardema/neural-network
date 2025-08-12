class_name NeuralNetwork

const Neuron = preload("uid://bsmeer8twyv7v")
const Layer = preload("uid://og5tgmr4jdw3")

signal updated
signal propagated_forwards
signal trained_frame

const Math = preload('Math.gd')
static var max_threads: int = 8

var data: TrainingData
var layers: Array[Layer]
var input_layer: Layer
var output_layer: Layer
var layer_count: int
var loss: float = 0
var config: NeuralNetworkConfig
var layout: Layout
var activation_functions: Activations
var loss_function: Utils.Math.Mean
var trainers: Array[Trainer]
var thread_count: int = 1
var scaling: int = 1
var stopwatch := Utils.Stopwatch.new()
#region Initialization

func _init(network_layout: Layout, network_config: NeuralNetworkConfig):
	activation_functions = Activations.new()
	layout = network_layout
	config = network_config
	loss_function = Utils.Math.Mean.new(Utils.Math.Mean.Type.MeanSquare)

	_build_network()

	for i in max_threads:
		trainers.append(Trainer.new(self , layout))


func destroy():
	data = null
	layers.clear()
	input_layer = null
	output_layer = null
	config = null
	layout = null
	activation_functions = null
	loss_function = null
	trainers.clear()


func _build_network():
	layer_count = layout.hidden.size() + 1

	# Setup layers and populate neurons
	for l in range(-1, layer_count):
		var layer := Layer.new()
		if l == -1:
			layer.size = layout.inputs
			input_layer = layer
		else:
			if l == layer_count - 1:
				layer.size = layout.outputs
				output_layer = layer
				layer.is_output = true
			else:
				layer.size = layout.hidden[l]
			layers.append(layer)

		for n in layer.size:
			layer.neurons.append(Neuron.new())

	for l in range(-1, layer_count):
		var layer := input_layer if l == -1 else layers[l]
		if l > 0: layer.previous = layers[l - 1]
		if l < layer_count - 1: layer.next = layers[l + 1]

		for n in layer.size:
			var neuron := layer.neurons[n]
			neuron.neuron_idx = n
			neuron.incoming = layer.previous.neurons if layer.previous else input_layer.neurons
			if layer.next: neuron.outgoing = layer.next.neurons
			neuron.incoming_size = neuron.incoming.size()
			neuron.outgoing_size = neuron.outgoing.size()
			neuron.layer = layer
			if layer.is_output: neuron.activation_type = layout.output_activation
			else: neuron.activation_type = layout.hidden_activation
			neuron.activation_function = activation_functions.get_function(neuron.activation_type)
			neuron.initialize(config.wt_std_dev)


func reset():
	data.rewind()
	loss = 1
	initialize_weights()


func initialize_weights():
	for layer in layers:
		for neuron in layer.neurons:
			neuron.initialize(config.wt_std_dev)
	updated.emit()

#endregion


#region Training

func _apply_learning_rate(rate: float):
	for layer in layers:
		for neuron in layer.neurons:
			neuron.learning_rate = rate


func train(iterations := config.iterations):
	loss = 0
	data.batch_size = config.batch_size

	if layers[0].neurons[0].learning_rate != config.learn_rate:
		_apply_learning_rate(config.learn_rate)

	var samples_per_frame: int = config.iterations
	var samples_per_batch: int = min(samples_per_frame, data.batch_size, data.samples)
	var batches_per_frame: int = max(1, samples_per_frame / samples_per_batch)
	var tasks_per_batch: int = min(trainers.size(), samples_per_batch, scaling)
	var samples_per_task: int = max(1, samples_per_batch / tasks_per_batch)

	# stopwatch.start()

	for batch_id in batches_per_frame:
		var batch_sample_index := data.current_index
		var group_task_id := WorkerThreadPool.add_group_task(
			train_task_kernel.bind(batch_sample_index, samples_per_task),
			tasks_per_batch,
			-1 if thread_count >= max_threads else thread_count,
			true
		)
		WorkerThreadPool.wait_for_group_task_completion(group_task_id)
		# stopwatch.lap('batch_%d' % batch_id)
		# for task_id in tasks_per_batch:
		# 	train_task_kernel(task_id, batch_sample_index, samples_per_task)

		data.next_sample(data.shuffle_training, samples_per_batch)
		if data.current_index % data.batch_size == 0:
			apply_parameters_from_gradients(data.batch_size)
	loss = loss_function.calculate()
	updated.emit()
	trained_frame.emit()

	#region Debug
	var process_time: float = Performance.get_monitor(Performance.Monitor.TIME_PROCESS)
	DebugTools.write('FPS', '%d' % Engine.get_frames_per_second(), false)
	DebugTools.write('Process', '%.2fms' % (process_time * 1000), false)

	# DebugTools.write('Current index', '%d' % data.current_index, false)
	# DebugTools.write('Current sample', '%d' % data.current_sample, false)
	# DebugTools.write('Samples', '%d' % samples_per_frame, false)
	# DebugTools.write('Batches', '%d' % batches_per_frame, false)
	# DebugTools.write('Batch samples', '%d' % samples_per_batch, false)
	# DebugTools.write('Batch tasks', '%d' % tasks_per_batch, false)
	# DebugTools.write('Batch task samples', '%d' % samples_per_task, false)

	# for ts in stopwatch.timestamps:
	# 	DebugTools.write('STS ' + ts, stopwatch.timestamps[ts], true, 4)
	# for lap in stopwatch.laps:
	# 	DebugTools.write('SLP ' + lap, stopwatch.laps[lap], true, 4)
	#endregion


func train_task_kernel(task_id: int, batch_sample_index: int, samples_per_task: int):
	for s in samples_per_task:
		var task_sample_index: int = batch_sample_index + samples_per_task * task_id + s
		trainers[task_id].batch_pass(task_sample_index)
		loss_function.append_array(trainers[task_id].deltas[-1])
		# if task_id == 0: stopwatch.lap('tk_%d_%d' % [task_id, s])
	# stopwatch.timestamp('task_%d' % task_id)


func propagate_forwards():
	for i in input_layer.size:
		input_layer.neurons[i].activation = data.get_input(i)

	for layer in layers:
		for neuron in layer.neurons:
			neuron.compute_activation()

	propagated_forwards.emit()


func apply_parameters_from_gradients(sum_size: int = 1):
	for layer in layers:
		for neuron in layer.neurons:
			neuron.average_bias_gradient(sum_size)
			neuron.average_weight_gradients(sum_size)
			neuron.update_weights_from_gradients()
			neuron.update_bias_from_gradient()
			neuron.reset_gradients()

#endregion


#region Helpers


#endregion


#region Sub Classes

class Trainer:
	var nn: NeuralNetwork
	var layout: Layout
	var activations: Array[PackedFloat32Array]
	var deltas: Array[PackedFloat32Array]
	var sample_idx: int


	func _init(network: NeuralNetwork, network_layout: Layout):
		nn = network
		layout = network_layout
		var input_data := PackedFloat32Array()
		input_data.resize(layout.inputs)

		for layer in nn.layers:
			var layer_activations := PackedFloat32Array()
			var layer_deltas := PackedFloat32Array()
			layer_activations.resize(layer.size)
			layer_deltas.resize(layer.size)
			activations.append(layer_activations)
			deltas.append(layer_deltas)
		activations.append(input_data)


	func batch_pass(sample_index: int):
		if sample_index >= nn.data.samples:
			push_warning('Sample index %d out of range' % sample_index)
			return
		sample_idx = sample_index

		for i in layout.inputs:
			activations[-1][i] = nn.data.get_input(i, sample_idx)

		# Forward activation
		for l in nn.layer_count:
			var layer := nn.layers[l]
			for n in layer.size:
				var neuron := layer.neurons[n]
				var pre := neuron.bias
				for i in neuron.incoming_size:
					pre += activations[l - 1][i] * neuron.weights[i]
				activations[l][n] = neuron.activation_function.activate(pre)

		# Compute deltas
		for l in range(nn.layer_count - 1, -1, -1):
			var layer := nn.layers[l]
			for n in layer.size:
				var neuron := layer.neurons[n]
				deltas[l][n] = 0
				if layer == nn.output_layer:
					deltas[l][n] = nn.data.get_target(n, sample_idx) - activations[l][n]
				else:
					for o in neuron.outgoing_size:
						deltas[l][n] += deltas[l + 1][o] * neuron.outgoing[o].weights[n]
				var diff: float = neuron.activation_function.differentiate(activations[l][n])
				deltas[l][n] *= diff

				# Increment gradients
				neuron.bias_gradient += deltas[l][n]
				for w in neuron.incoming_size:
					neuron.weight_gradients[w] += activations[l - 1][w] * deltas[l][n]


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
