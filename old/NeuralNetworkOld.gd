@abstract
class_name NeuralNetworkOld

signal network_updated

var layer_sizes: Array[int]

var inputs_count: int
var outputs_count: int
var neurons_count: int
var weights_count: int
var layers_count: int

var weight_offsets: Array[int]
var offsets: Array[int]

var weights: PackedFloat32Array
var biases: PackedFloat32Array
var activations: PackedFloat32Array
var errors: PackedFloat32Array
var training_data: PackedFloat32Array
var target_outputs: PackedFloat32Array

var current_sample: int = 0

func _init(inputs: int, hidden_layout: Array[int], outputs: int):
	inputs_count = inputs
	outputs_count = outputs
	layers_count = hidden_layout.size() + 2
	layer_sizes.append_array([inputs] + hidden_layout + [outputs])

	weight_offsets.resize(layers_count)
	offsets.resize(layers_count)
	neurons_count = layer_sizes.reduce(func(s, l): return s + l)

	calc_layout()

	weights.resize(weights_count)
	biases.resize(neurons_count)
	activations.resize(neurons_count)
	errors.resize(neurons_count)

	training_data.resize(inputs_count)
	target_outputs.resize(outputs_count)

func calc_layout():
	var offset := 0
	for l in layers_count:
		offsets[l] = offset
		weight_offsets[l] = weights_count
		offset += layer_sizes[l]
		if l > 0: weights_count += layer_sizes[l - 1] * layer_sizes[l]

func initialize():
	initialize_weights()

func initialize_weights():
	for n in neurons_count:
		activations[n] = 0
		biases[n] = 0
		errors[n] = 0
	for w in weights_count:
		#weights[w] = 1
		weights[w] = rand_normal(1)
		#weights[w] = float(w - weights_count / 2) / weights_count * 2

	network_updated.emit()

func set_training_data(training: PackedFloat32Array, target: PackedFloat32Array):
	training_data = training
	target_outputs = target

static func normal_distribution(x: float, mean: float, sd: float):
	return (PI * sd) * exp(-0.5 * ((x - mean) / sd) ** 2)

static func rand_normal(std_dev: float = 0.01):
	var u1 = randf()
	var u2 = randf()

	return sqrt(-2.0 * log(max(u1, 1e-7))) * cos(2.0 * PI * u2) * std_dev

func get_weight(layer: int, node: int, prev_node: int) -> float:
	if layer == 0: return 0.0

	var index := weight_offsets[layer - 1] + node * layer_sizes[layer - 1] + prev_node

	return weights[index]

@abstract
func train(iterations: int)

func debug():
	#printt("layers", "%2d" % layers_count)
	#printt("inputs", "%2d" % inputs_count)
	#printt("hidden", "%2d" % (neurons_count - inputs_count - outputs_count))
	#printt("outputs", "%2d" % outputs_count)
	#printt("neurons", "%2d" % neurons_count)
	#printt("weights", "%2d" % weights_count)
	#printt("biases", "%2d" % biases.size())
	#printt("activ.", "%2d" % activations.size())
	#printt("errors", "%2d" % errors.size())
	#print(weights)
	print(activations)
	print(biases)
	#print(errors)
	#prints("weight_offsets:", weight_offsets)
	#prints("offsets:", offsets)
	#prints("training_data:", training_data)
	#printt(weights[0], activations[1], biases[0], errors[0])
	print("-".repeat(10))
