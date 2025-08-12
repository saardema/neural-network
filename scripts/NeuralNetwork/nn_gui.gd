extends Control

var nn: NeuralNetwork
const NetworkManager = preload("uid://b85gq2coeewlj")
const Swiper = preload("uid://c8lh45ojaj4dv")

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
@onready var button_next_sample: Button = %ButtonNextSample
@onready var button_restart: Button = %ButtonRestart

@onready var input_batch_size: Swiper = %InputBatchSize
@onready var input_samples_per_frame: Swiper = %InputSamplesPerFrame
@onready var input_wt_std_dev: Swiper = %InputWeightsInit
@onready var input_random_sampling: CheckButton = %InputRandomSampling
@onready var input_sampling_resolution: HSlider = %InputSamplingResolution
@onready var input_shuffle_training: CheckButton = %InputShuffleTraining
@onready var input_sample_count: Swiper = %InputSampleCount
@onready var input_learn_rate: Swiper = %InputLearnRate
@onready var input_thread_count: Swiper = %InputThreadCount
@onready var input_batch_ratio: Swiper = %InputBatchRatio

@onready var value_sampling_resolution: Label = %ValueSamplingResolution

signal problem_changed()

var save_rate_limiter: RateLimiter
var channels: Dictionary[StringName, Plotter.Channel]

const CH_LOSS = "Loss"
const CH_INPUT = "Input"
const CH_ERROR = "Error"
const CH_OUTPUT = "Output"
const CH_EXPECTED = "Expected"
const CH_SELECTED = "Selected"

func _ready():
	input_batch_size.value_changed.connect(update_config)
	input_samples_per_frame.value_changed.connect(update_config)
	input_learn_rate.value_changed.connect(update_config)
	input_wt_std_dev.value_changed.connect(update_config)
	input_sample_count.value_changed.connect(update_config)

	input_sampling_resolution.value_changed.connect(
		func(v): value_sampling_resolution.text = str(int(v)))
	value_sampling_resolution.text = str(int(input_sampling_resolution.value))

	button_fwd.pressed.connect(_on_fwd_pressed)
	button_next_sample.pressed.connect(_on_next_sample_pressed)
	button_reset.pressed.connect(_on_reset_pressed)
	button_restart.pressed.connect(_on_reset_pressed)
	input_shuffle_training.toggled.connect(func(v): nn.data.shuffle_training = v)
	input_batch_ratio.value_changed.connect(func(v): nn.scaling = int(v))
	input_thread_count.value_changed.connect(func(v): nn.thread_count = int(v))
	button_step.pressed.connect(func(): nn.train(1))

	for problem_type in NetworkManager.Problem.Type:
		popup_problems.add_check_item(problem_type)
	popup_problems.connect("id_pressed", _on_problem_selected)

	# Plotters
	channels[CH_INPUT] = io_plotter.create_channel(CH_INPUT)
	channels[CH_ERROR] = io_plotter.create_channel(CH_ERROR)
	channels[CH_OUTPUT] = io_plotter.create_channel(CH_OUTPUT)
	channels[CH_EXPECTED] = io_plotter.create_channel(CH_EXPECTED)
	channels[CH_SELECTED] = io_plotter.create_channel(CH_SELECTED)

	channels[CH_LOSS] = loss_plotter.create_channel(CH_LOSS)
	loss_plotter.create_channel('FPS')

	save_rate_limiter = RateLimiter.new(func(): nn.config.save(), 2, RateLimiter.Mode.CONCLUDE)


func set_network(network: NeuralNetwork):
	if nn != null:
		nn.destroy()
		nn.init_ref()
		nn.unreference()

	nn = network
	network_display.set_network(nn)

	nn.updated.connect(render)
	nn.trained_frame.connect(sample_output)

	nn.data.shuffle_training = input_shuffle_training.button_pressed
	nn.scaling = int(input_batch_ratio.value)
	nn.thread_count = int(input_thread_count.value)

	input_batch_size.set_value_no_signal(nn.config.batch_size)
	input_samples_per_frame.set_value_no_signal(nn.config.iterations)
	input_learn_rate.set_value_no_signal(nn.config.learn_rate)
	input_wt_std_dev.set_value_no_signal(nn.config.wt_std_dev)
	input_sample_count.set_value_no_signal(nn.config.sample_count)

	channels[CH_INPUT].set_channel_count(nn.input_layer.size)
	channels[CH_OUTPUT].set_channel_count(nn.output_layer.size)
	channels[CH_ERROR].set_channel_count(nn.output_layer.size)
	channels[CH_EXPECTED].set_channel_count(nn.output_layer.size)

	for i in popup_problems.item_count:
		var problem_name := popup_problems.get_item_text(i)
		popup_problems.set_item_checked(i, problem_name == nn.config.problem)


	render()


func _on_reset_pressed():
	io_plotter.clear()
	loss_plotter.clear()


func _on_problem_selected(problem_id: int) -> void:
	nn.config.problem = popup_problems.get_item_text(problem_id)
	nn.config.save()
	problem_changed.emit()


func sample_output():
	var resolution := int(input_sampling_resolution.value)
	var original_index := nn.data.current_index
	nn.data.current_index = 0
	channels[CH_INPUT].flush()
	channels[CH_OUTPUT].flush()
	channels[CH_EXPECTED].flush()
	channels[CH_SELECTED].flush()

	for i in nn.data.get_sampling_range(resolution):
		nn.data.set_sample(i, input_random_sampling.button_pressed)
		nn.propagate_forwards()
		channels[CH_INPUT].write_singles(nn.input_layer.as_array())
		channels[CH_OUTPUT].write_singles(nn.output_layer.as_array())
		channels[CH_EXPECTED].write_singles(nn.data.get_targets())
		if network_display.selected_neuron:
			channels[CH_SELECTED].write_single(network_display.selected_neuron.activation)

	channels[CH_LOSS].write_single(nn.loss)
	channels[CH_ERROR].write_singles(nn.output_layer.as_delta_array())
	loss_plotter.channels['FPS'].write_single(Engine.get_frames_per_second())

	nn.data.current_index = original_index
	nn.data.current_sample = nn.data.indices[original_index]


func render():
	network_display.queue_redraw()
	queue_redraw()


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
	nn.config.iterations = int(input_samples_per_frame.value)
	nn.config.learn_rate = input_learn_rate.value
	nn.config.wt_std_dev = input_wt_std_dev.value
	nn.config.sample_count = int(input_sample_count.value)

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
	left_text_label.text += "\nLoss:     %.6f" % nn.loss
	left_text_label.text += "\nFPS:      %d" % Engine.get_frames_per_second()
