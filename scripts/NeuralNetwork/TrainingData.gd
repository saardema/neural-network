class_name TrainingData

signal batch_finished
signal epoch_finished

var training_data: PackedFloat32Array
var target_data: PackedFloat32Array
var current_sample: int
var current_index: int
var inputs: int
var outputs: int
var samples: int
var indices: PackedInt32Array
var _last_sampling_range: Array[int]
var shuffle_training: bool = true
var batch_size: int = 1:
	set(v):
		batch_size = _set_batch_size(v)

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


func _set_batch_size(value: int) -> int:
	value = clamp(value, 1, samples)
	if value != batch_size:
		rewind()

	return value


func rewind():
	current_sample = 0
	current_index = 0


func get_sampling_range(resolution: int) -> Array[int]:
	resolution = min(resolution, samples)
	if resolution == _last_sampling_range.size() - 1 and resolution <= samples:
		return _last_sampling_range

	var sampling_range: Array[int]

	for i in resolution:
		sampling_range.append(floori(i * samples / resolution))
	sampling_range.append(samples - 1)

	_last_sampling_range = sampling_range

	return sampling_range


func shuffle_indices():
	for i in samples - 1:
		var swap_idx: int = randi_range(i, samples - 1)
		var index: int = indices[swap_idx]
		indices[swap_idx] = indices[i]
		indices[i] = index


func next_sample(shuffle: bool = shuffle_training, step := 1):
	current_index += step

	if current_index % batch_size == 0:
		batch_finished.emit()

	if current_index >= samples:
		epoch_finished.emit()
		rewind()
		if shuffle: shuffle_indices()

	current_sample = indices[current_index] if shuffle else current_index


func set_sample(index: int, shuffle: bool = shuffle_training):
	current_index = index
	current_sample = indices[current_index] if shuffle else current_index


func get_input(input_idx: int, index := -1) -> float:
	if index == -1:
		index = current_sample
	elif shuffle_training:
		index = indices[index]

	return training_data[index * inputs + input_idx]


func get_target(output_idx: int, index := -1) -> float:
	if index == -1:
		index = current_sample
	elif shuffle_training:
		index = indices[index]

	return target_data[index * outputs + output_idx]


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
