class_name TrainingData
extends RefCounted

var training_data: PackedFloat32Array
var target_data: PackedFloat32Array
var current_sample: int
var current_index: int
var inputs: int
var outputs: int
var samples: int
var indices: PackedInt32Array
var _last_sample_resolution: int
var _last_sampling_range: Array[int]


func _init(
		input_data: PackedFloat32Array,
		expected_output: PackedFloat32Array,
		input_count: int,
		output_count: int,
		sample_count: int
	):
	training_data = input_data
	target_data = expected_output
	inputs = input_count
	outputs = output_count
	samples = sample_count
	indices.resize(samples)
	for i in samples: indices[i] = i
	shuffle_indices()

	if training_data.size() % inputs != 0:
		push_error("Training data length is not a multiple of the number of inputs")

	if target_data.size() % outputs != 0:
		push_error("Target data length is not a multiple of the number of outputs")


func get_sampling_range(resolution: int) -> Array[int]:
	if resolution == _last_sample_resolution:
		return _last_sampling_range
	DebugTools.print("Sampling range for resolution %d" % resolution)
	var rng: Array[int]

	for i in float(resolution):
		rng.append(floori(i * samples / resolution))
	rng.append(samples - 1)

	_last_sample_resolution = resolution
	_last_sampling_range = rng

	return rng


func shuffle_indices():
	DebugTools.print('Shuffling indices')
	for i in samples - 1:
		var swap_idx: int = randi_range(i, samples - 1)
		var index: int = indices[swap_idx]
		indices[swap_idx] = indices[i]
		indices[i] = index


func next_sample(shuffle: bool = true, step := 1):
	current_index += step

	if current_index >= samples:
		current_index = 0
		if shuffle:
			shuffle_indices()

	current_sample = indices[current_index] if shuffle else current_index


func get_input(input_idx: int) -> float:
	return training_data[current_sample * inputs + input_idx]


func get_target(output_idx: int) -> float:
	if target_data.size() == 0: return 0
	return target_data[current_sample * outputs + output_idx]


func get_inputs() -> PackedFloat32Array:
	return training_data.slice(
		current_sample * inputs,
		current_sample * inputs + inputs
	)

func get_targets() -> PackedFloat32Array:
	return target_data.slice(
		current_sample * outputs,
		current_sample * outputs + outputs
	)
