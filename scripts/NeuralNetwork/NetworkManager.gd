extends Control

@onready var nn_gui: Control = $NNGUI
var nn: NeuralNetwork

var training_data: PackedFloat32Array
var control_data: PackedFloat32Array
@export var config: NeuralNetworkConfig


func _ready():
	init_network()
	nn_gui.button_reload.pressed.connect(init_network)
	nn_gui.problem_changed.connect(init_network)

func init_network():
	var problem := Problem.create(config.problem, 10000)
	nn = NeuralNetwork.new(problem.layout, config)
	nn.set_training_data(problem.training_data)
	nn_gui.set_network(nn)
	nn_gui.button_reset.pressed.connect(nn.reset)


func _process(_dt: float):
	if nn_gui.button_run.button_pressed:
		nn.train()


@abstract class Problem:
	var layout := NeuralNetwork.Layout.new()
	var input_data := PackedFloat32Array()
	var target_data := PackedFloat32Array()
	var output_activation := Activations.Type.Sigmoid
	var training_data: TrainingData
	var samples: int

	static func create(problem_type: String, samples_count := 100) -> Problem:
		return Type.get(problem_type, Basic).new(samples_count)

	static var Type: Dictionary = {
		"Basic": Basic,
		"BinaryOut": BinaryOut,
		"Xor": Xor,
		"Sin": Sin,
		"Polynomial": Polynomial,
		"CircleClassification": CircleClassification
	}

	var _in_ids: Array[int]
	var _out_ids: Array[int]

	func init(): pass
	func _input_gen(_sample: int) -> Array[float]: return []
	func _target_gen(_sample: int) -> Array[float]: return []

	func _norm_x(x: int) -> float:
		return float(x) / float(samples - 1)

	func _init(sample_count: int):
		samples = sample_count
		init()
		for i in layout.inputs: _in_ids.append(i)
		for i in layout.outputs: _out_ids.append(i)
		training_data = generate_training_data()

	func generate_training_data() -> TrainingData:
		if self.has_method('_combined_gen'):
			self.call('_combined_gen')
		elif self.has_method('_input_array_gen'):
			self.call('_input_array_gen')
			self.call('_output_array_gen')
		else:
			for i in samples:
				input_data.append_array(_input_gen(i))
				target_data.append_array(_target_gen(i))

		return TrainingData.new(input_data, target_data, layout.inputs, layout.outputs, samples)

class Basic extends Problem:
	func init(): layout.outputs = 2
	func _input_gen(x): return [_norm_x(x)]
	func _target_gen(x): return [0, 0] if x < 2 else [1, 1]

class BinaryOut extends Problem:
	func init():
		layout.outputs = 3
		layout.hidden = [3, 4]
		layout.output_activation = Activations.Type.LeakyReLu

	func _input_gen(x: int):
		return [float(x % 8) / 3.5 - 1.0]

	func _target_gen(x: int):
		var v = int((x + 1.0) * 3.5)
		return _out_ids.map(func(i): return float((v >> i) & 1))

class Xor extends Problem:
	func init():
		layout.output_activation = Activations.Type.LeakyReLu
		layout = NeuralNetwork.Layout.new(2, [4, 4], 1)

	func _input_gen(x: int) -> Array[float]:
		return [(x % 4) / 2, x % 2]

	func _target_gen(x: int):
		return [1 if x % 4 in [1, 2] else 0]

class Sin extends Problem:
	func init():
		layout.hidden = [3, 4, 2]
		layout.output_activation = Activations.Type.PreLu

	func _input_gen(x: int):
		return [_norm_x(x) * 2 - 1]

	func _target_gen(x: int):
		return [1.8 - sin(_norm_x(x) * TAU)]

class Polynomial extends Problem:
	var max_value: float = 1

	func init():
		max_value = _target_gen(samples - 1)[0]
		layout.hidden = [7, 7, 4]
		layout.output_activation = Activations.Type.SoftMax

	func _input_gen(x: int):
		return [_norm_x(x)]

	func _target_gen(x: int):
		var v: float = x + 5 * x * x - (10 * x * x * x)

		return [v / max_value]

class CircleClassification extends Problem:
	func init():
		layout.inputs = 2
		layout.hidden = [3, 3]
		layout.output_activation = Activations.Type.TanH

	func _combined_gen():
		var x: float = 0
		var y: float = 0

		for s in samples:
			x = randf_range(-2, 2)
			y = randf_range(-2, 2)
			input_data.append_array([x, y])

			var target := 1 if (x * x + y * y) < 1.5 else 0
			target_data.append(target)
