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

signal problem_changed()

var next_sample_is_random: bool
var save_rate_limiter: RateLimiter
var channels: Dictionary[StringName, Plotter.Channel]

const LOSS = "Loss"
const IO_INPUT = "Input"
const IO_OUTPUT = "Output"
const IO_EXPECTED = "Expected"

func _ready():
	input_batch_size.value_changed.connect(update_config)
	input_iterations.value_changed.connect(update_config)
	input_learn_rate.value_changed.connect(update_config)
	input_wt_std_dev.value_changed.connect(update_config)

	button_fwd.pressed.connect(_on_fwd_pressed)
	button_next_sample.pressed.connect(_on_next_sample_pressed)
	button_reset.pressed.connect(_on_reset_pressed)
	button_reload.pressed.connect(_on_reset_pressed)

	for problem_type in NetworkManager.Problem.Type:
		popup_problems.add_item(problem_type)
	popup_problems.connect("id_pressed", _on_problem_selected)

	# Plotters
	channels[LOSS] = loss_plotter.create_channel(LOSS)
	channels[IO_INPUT] = io_plotter.create_channel(IO_INPUT)
	channels[IO_OUTPUT] = io_plotter.create_channel(IO_OUTPUT)
	channels[IO_EXPECTED] = io_plotter.create_channel(IO_EXPECTED)

	loss_plotter.add_channel(channels[IO_INPUT])
	loss_plotter.add_channel(channels[IO_OUTPUT])
	loss_plotter.add_channel(channels[IO_EXPECTED])


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
	nn.network_updated.connect(render)
	nn.propagated_forwards.connect(_on_forward_propagation)
	network_display.set_network(nn)

	button_bwd.pressed.connect(nn.step_bwd)
	button_step.pressed.connect(nn.train)

	input_batch_size.set_value_no_signal(nn.config.batch_size)
	input_iterations.set_value_no_signal(nn.config.iterations)
	input_learn_rate.set_value_no_signal(nn.config.learn_rate)
	input_wt_std_dev.set_value_no_signal(nn.config.wt_std_dev)

	channels[IO_INPUT].set_channel_count(nn.input_layer.size)
	channels[IO_OUTPUT].set_channel_count(nn.output_layer.size)
	channels[IO_EXPECTED].set_channel_count(nn.output_layer.size)

	input_learn_rate.value = nn.config.learn_rate
	value_learn_rate.text = &"%.3f" % input_learn_rate.value

	save_rate_limiter = RateLimiter.new(nn.config.save, 2, RateLimiter.Mode.CONCLUDE)

	render()


func _on_forward_propagation():
	if nn.input_layer.size == 1:
		io_plotter.write_channel(channels[IO_INPUT], nn.input_layer.neurons[0].activation)
	else:
		io_plotter.write_multi_channel(channels[IO_INPUT], nn.input_layer.as_array())

	if nn.output_layer.size == 1:
		io_plotter.write_channel(channels[IO_OUTPUT], nn.output_layer.neurons[0].activation)
		io_plotter.write_channel(channels[IO_EXPECTED], nn.data.target_data[nn.data.current_sample])
	else:
		io_plotter.write_multi_channel(channels[IO_OUTPUT], nn.output_layer.as_array())


func render():
	loss_plotter.write_channel(channels[LOSS], nn.loss)
	network_display.queue_redraw()
	queue_redraw()

func _on_fwd_pressed():
	nn.next_sample()
	nn.step_fwd()

func _on_next_sample_pressed():
	nn.next_sample(true)
	nn.step_fwd()

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
	var expected: Array[String]
	var output: Array[String]
	for o in nn.output_layer.size:
		expected.append("%5.2f" % nn.data.get_target(o))
		output.append("%5.2f" % nn.output_layer.neurons[o].activation)

	left_text_label.text = nn.config.problem
	left_text_label.text += "\nSample: %d" % nn.data.current_sample
	left_text_label.text += "\nExpected: [%s]" % ', '.join(expected)
	left_text_label.text += "\nOutput:   [%s]" % ', '.join(output)
	left_text_label.text += "\nLoss: %.6f" % nn.loss
