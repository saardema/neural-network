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
	var problem := Problem.create(config.problem, 400)
	nn = NeuralNetwork.new(problem.layout, config)
	nn.data = problem.training_data
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

	func _normalize(x: int) -> float:
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
	func init():
		layout.inputs = 1
		layout.outputs = 1
		layout.hidden = [10, 10]
		layout.hidden_activation = Activations.Type.PreLu
		layout.output_activation = Activations.Type.PreLu

	func _combined_gen():
		var corners := []
		var cidx: int = 0
		var dist: float = 0
		var steps: int = 5
		for i in steps + 1:
			corners.append(Vector2(dist, randf_range(0.1, 0.9)))
			dist += randf_range(0.5, 1.5) * (1.0 / steps)
		corners[-1].x = 1

		for s in samples:
			var x := float(s) / samples
			if x > corners[cidx + 1].x: cidx += 1
			var v1: Vector2 = corners[cidx]
			var v2: Vector2 = corners[cidx + 1]
			var t: float = inverse_lerp(v1.x, v2.x, x)
			input_data.append(x)
			# input_data.append_array([x, v1.x, v1.y, v2.x, v2.y])
			target_data.append(lerp(v1, v2, t).y)


class BinaryOut extends Problem:
	func init():
		layout.outputs = 3
		layout.hidden = [3, 4]
		layout.output_activation = Activations.Type.PreLu

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
		layout.hidden = [8]
		layout.hidden_activation = Activations.Type.TanH
		layout.output_activation = Activations.Type.TanH

	func _combined_gen():
		for s in samples:
			var x := float(s) / samples * 2 - 1
			input_data.append((x))
			target_data.append(sin(x * TAU) * 0.5)


class Polynomial extends Problem:
	var min_value: float
	var max_value: float
	var value_range: float
	var y: float

	func init():
		layout.inputs = 1
		layout.hidden = [8, 8]
		layout.output_activation = Activations.Type.SoftMax

	func f(x: float):
		return 0.3 * pow(x - 0.6, 5) - 1.5 * pow(x - 0.35, 3)

	func _combined_gen():
		for s in samples:
			var x := float(s) / samples * 2 - 1
			input_data.append(x)
			target_data.append(f(x))


class CircleClassification extends Problem:
	func init():
		layout.inputs = 2
		layout.hidden = [6, 6]
		layout.output_activation = Activations.Type.LeakyReLu

	func _combined_gen():
		var x: float = 0
		var y: float = 0

		for s in samples:
			x = randf_range(-2, 2)
			y = randf_range(-2, 2)
			input_data.append_array([x, y])

			var target := 1 if (x * x + y * y) < 2 else 0
			target_data.append(target)
