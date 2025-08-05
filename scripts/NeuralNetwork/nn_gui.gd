extends Control

var nn: NeuralNetwork
const NetworkManager = preload("uid://b85gq2coeewlj")

@onready var network_display: Control = %NetworkDisplay
@onready var io_plotter: Plotter = %IOPlotter
@onready var loss_plotter: Plotter = %LossPlotter
@onready var left_text_label: Label = %LeftTextLabel
@onready var menu_problems: MenuButton = %MenuProblems
@onready var popup_problems: PopupMenu = %MenuProblems.get_popup()

@onready var button_reset: Button = %ButtonReset
@onready var button_step: Button = %ButtonStep
@onready var button_run: Button = %ButtonRun
@onready var button_fwd: Button = %ButtonFWD
@onready var button_bwd: Button = %ButtonBWD
@onready var button_next_sample: Button = %ButtonNextSample
@onready var button_reload: Button = %ButtonReload

@onready var input_batch_size: SpinBox = %InputBatchSize
@onready var input_iterations: SpinBox = %InputIterations
@onready var input_learn_rate: HSlider = %InputLearnRate
@onready var input_wt_std_dev: SpinBox = %InputWeightsInit
@onready var value_learn_rate: Label = %ValueLearnRate
@onready var input_random_sampling: CheckButton = %InputRandomSampling
@onready var input_sampling_resolution: HSlider = %InputSamplingResolution
@onready var value_sampling_resolution: Label = %ValueSamplingResolution
@onready var input_shuffle_training: CheckButton = %InputShuffleTraining

signal problem_changed()

var save_rate_limiter: RateLimiter
var channels: Dictionary[StringName, Plotter.Channel]

const CH_LOSS = "Loss"
const CH_INPUT = "Input"
const CH_ERROR = "Error"
const CH_OUTPUT = "Output"
const CH_EXPECTED = "Expected"


func _ready():
	input_batch_size.value_changed.connect(update_config)
	input_iterations.value_changed.connect(update_config)
	input_learn_rate.value_changed.connect(update_config)
	input_wt_std_dev.value_changed.connect(update_config)

	input_sampling_resolution.value_changed.connect(
		func(v): value_sampling_resolution.text = str(int(v)))
	value_sampling_resolution.text = str(int(input_sampling_resolution.value))

	button_bwd.pressed.connect(_on_bwd_pressed)
	button_fwd.pressed.connect(_on_fwd_pressed)
	button_next_sample.pressed.connect(_on_next_sample_pressed)
	button_reset.pressed.connect(_on_reset_pressed)
	button_reload.pressed.connect(_on_reset_pressed)

	for problem_type in NetworkManager.Problem.Type:
		popup_problems.add_item(problem_type)
	popup_problems.connect("id_pressed", _on_problem_selected)

	# Plotters
	channels[CH_LOSS] = io_plotter.create_channel(CH_LOSS)
	channels[CH_INPUT] = io_plotter.create_channel(CH_INPUT)
	channels[CH_ERROR] = io_plotter.create_channel(CH_ERROR)
	channels[CH_OUTPUT] = io_plotter.create_channel(CH_OUTPUT)
	channels[CH_EXPECTED] = io_plotter.create_channel(CH_EXPECTED)
	loss_plotter.add_channel(channels[CH_LOSS])
	loss_plotter.add_channel(channels[CH_INPUT])
	loss_plotter.add_channel(channels[CH_ERROR])
	loss_plotter.add_channel(channels[CH_OUTPUT])
	loss_plotter.add_channel(channels[CH_EXPECTED])

func _on_reset_pressed():
	io_plotter.clear()
	loss_plotter.clear()


func _on_problem_selected(problem_id: int) -> void:
	var problem_type := popup_problems.get_item_text(problem_id)
	nn.config.problem = problem_type
	nn.config.save()
	problem_changed.emit()


func set_network(network: NeuralNetwork):
	nn = network
	network_display.set_network(nn)

	nn.updated.connect(render)
	nn.trained_frame.connect(sample_output)
	button_step.pressed.connect(nn.train.bind(1))

	input_batch_size.set_value_no_signal(nn.config.batch_size)
	input_iterations.set_value_no_signal(nn.config.iterations)
	input_learn_rate.set_value_no_signal(nn.config.learn_rate)
	input_wt_std_dev.set_value_no_signal(nn.config.wt_std_dev)

	channels[CH_INPUT].set_channel_count(nn.input_layer.size)
	channels[CH_OUTPUT].set_channel_count(nn.output_layer.size)
	channels[CH_ERROR].set_channel_count(nn.output_layer.size)
	channels[CH_EXPECTED].set_channel_count(nn.output_layer.size)

	nn.shuffle_training = input_shuffle_training.button_pressed
	input_shuffle_training.toggled.connect(
		func(is_toggled: bool): nn.shuffle_training = is_toggled
	)

	input_learn_rate.value = nn.config.learn_rate
	value_learn_rate.text = &"%.3f" % input_learn_rate.value

	save_rate_limiter = RateLimiter.new(nn.config.save, 2, RateLimiter.Mode.CONCLUDE)

	render()


func record_snapshot():
	if nn.input_layer.size == 1:
		channels[CH_INPUT].write_single(nn.input_layer.neurons[0].activation)
	else:
		channels[CH_INPUT].write_singles(nn.input_layer.as_array())

	if nn.output_layer.size == 1:
		channels[CH_OUTPUT].write_single(nn.output_layer.neurons[0].activation)
		channels[CH_EXPECTED].write_single(nn.data.target_data[nn.data.current_sample])
	else:
		channels[CH_OUTPUT].write_singles(nn.output_layer.as_array())
		channels[CH_EXPECTED].write_singles(nn.data.get_targets())


func sample_output():
	var resolution := int(input_sampling_resolution.value)
	var original_index := nn.data.current_index
	var random_sampling := input_random_sampling.button_pressed
	nn.data.current_index = 0
	var sampling_range := nn.data.get_sampling_range(resolution)

	for i in sampling_range:
		nn.data.current_sample = nn.data.indices[i] if random_sampling else i
		nn.propagate_forwards()
		record_snapshot()

	nn.data.current_index = original_index
	nn.data.current_sample = nn.data.indices[original_index]

	channels[CH_LOSS].write_single(nn.loss)
	channels[CH_ERROR].write_singles(nn.output_layer.as_delta_array())

func render():
	network_display.queue_redraw()
	queue_redraw()


func _on_bwd_pressed():
	nn.propagate_backwards()
	nn.updated.emit()


func _on_fwd_pressed():
	nn.data.next_sample(false)
	nn.propagate_forwards()
	nn.updated.emit()


func _on_next_sample_pressed():
	nn.data.next_sample(true)
	nn.propagate_forwards()
	nn.updated.emit()


func update_config(_v):
	nn.config.batch_size = int(input_batch_size.value)
	nn.config.iterations = int(input_iterations.value)
	nn.config.learn_rate = input_learn_rate.value
	nn.config.wt_std_dev = input_wt_std_dev.value

	value_learn_rate.text = &"%.3f" % input_learn_rate.value
	save_rate_limiter.exec()


static func array_to_str(arr: PackedFloat32Array) -> String:
	if arr.size() == 0: return ""

	var s := "["
	for f in arr: s += "%.2f, " % f
	s[-2] = "]"

	return s


func _draw():
	if not left_text_label.get_parent().visible: return

	var input: Array[String]
	var expected: Array[String]
	var output: Array[String]
	for i in nn.input_layer.size:
		input.append("%5.2f" % nn.input_layer.neurons[i].activation)

	for o in nn.output_layer.size:
		expected.append("%5.2f" % nn.data.get_target(o))
		output.append("%5.2f" % nn.output_layer.neurons[o].activation)

	left_text_label.text = nn.config.problem
	left_text_label.text += "\nSample: %d" % nn.data.current_index
	left_text_label.text += "\nInput:    [%s]" % ', '.join(input)
	left_text_label.text += "\nExpected: [%s]" % ', '.join(expected)
	left_text_label.text += "\nOutput:   [%s]" % ', '.join(output)
	left_text_label.text += "\nError:    [%5.2f]" % nn.output_layer.neurons[0].delta
	left_text_label.text += "\nLoss: %.6f" % nn.loss
