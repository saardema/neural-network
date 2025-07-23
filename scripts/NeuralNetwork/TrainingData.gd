class_name TrainingData
extends RefCounted

var training_data: PackedFloat32Array
var target_data: PackedFloat32Array
var current_sample: int
var inputs: int
var outputs: int
var samples: int

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

	if training_data.size() % inputs != 0:
		push_error("Training data length is not a multiple of the number of inputs")

	if target_data.size() % outputs != 0:
		push_error("Control data length is not a multiple of the number of outputs")

func next_sample(shuffle: bool = true, step := 1):
	if shuffle:
		current_sample = randi_range(0, samples - 1)
	else:
		current_sample = (current_sample + step) % samples

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
