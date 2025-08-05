class_name Layer

var neurons: Array[Neuron]
var size: int
var previous: Layer
var next: Layer
var is_output: bool


func as_scalar(normalize := true) -> float:
	var sum := 0.0
	for neuron in neurons:
		sum += neuron.activation
	if normalize: sum /= neurons.size()

	return sum

func as_array(scale := 1.0) -> PackedFloat32Array:
	var result := PackedFloat32Array()

	for neuron in neurons:
		var value := neuron.activation
		if scale != 1.0: value *= scale
		result.append(value)

	return result


func as_delta_array() -> PackedFloat32Array:
	var result := PackedFloat32Array()

	for neuron in neurons:
		result.append(neuron.delta)

	return result
