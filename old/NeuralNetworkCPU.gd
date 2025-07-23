class_name NeuralNetworkCPU
extends NeuralNetworkOld

var layers: Array[NeuralNetworkLayer]

func initialize():
	super.initialize()

	for l in range(1, layers_count):
		layers.append(NeuralNetworkLayer.new(l, layer_sizes[l - 1], layer_sizes[l], self ))

func train(iterations: int = 1):
	for i in iterations:
		forward_pass()
		current_sample = (current_sample + 1) % (training_data.size() / inputs_count)
	network_updated.emit()

func forward_pass():
	var training_idx := current_sample * inputs_count
	var input_values := training_data.slice(training_idx, training_idx + inputs_count)

	for i in inputs_count: activations[i] = input_values[i]
	for layer in layers: input_values = layer.get_outputs(input_values)

# func backward_pass():
# 	var expected_outputs

class NeuralNetworkLayer:
	var inputs: int
	var outputs: int
	var layer_idx: int
	var nn: NeuralNetworkOld

	func _init(layer: int, ins: int, outs: int, network: NeuralNetworkOld):
		layer_idx = layer
		inputs = ins
		outputs = outs
		nn = network

	static func sigmoid(x: float) -> float:
		return 1.0 / (1.0 + exp(-x))

	static func sigmoid_derivative(x: float) -> float:
		var s := sigmoid(x)

		return s * (1.0 - s)

	func get_bias(output_idx: int) -> float:
		return nn.biases[nn.offsets[layer_idx] + output_idx]

	func calc_errors(next_errors: PackedFloat32Array) -> PackedFloat32Array:
		var errors: PackedFloat32Array
		errors.resize(outputs)

		for o in outputs:
			var error := 0.0

			for n in nn.layer_sizes[layer_idx + 1]:
				error += nn.get_weight(layer_idx + 1, n, o)

			nn.errors[nn.offsets[layer_idx] + o] = error

		return errors

	func get_outputs(input_values: PackedFloat32Array) -> PackedFloat32Array:
		var output_values: PackedFloat32Array
		output_values.resize(outputs)

		for o in outputs:
			output_values[o] = get_bias(o)
			var weight_idx := nn.weight_offsets[layer_idx - 1]
			for i in input_values.size():
				output_values[o] += input_values[i] * nn.weights[weight_idx + i]
			output_values[o] = sigmoid(output_values[o])

			nn.activations[nn.offsets[layer_idx] + o] = output_values[o]

		return output_values
