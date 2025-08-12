extends Control

@onready var plotter: Plotter = $Plotter
var channel: Plotter.Channel
var plot_scheduled: bool = true
var samples: float = 200
@export var x_range: int = 1:
	set(v):
		x_range = v
		plot()

@export var y_offset: float = 0:
	set(v):
		y_offset = v
		plot()

@export var terms: PackedVector3Array:
	set(v):
		terms = v
		plot()

@export_range(0, 1) var resolution: float = 1:
	set(v):
		resolution = v
		samples = ceilf(200.0 * resolution)
		plot()


func _ready():
	channel = plotter.create_channel('Curve', 2)

func smoothstep2(t):
	return t ** terms[0].x * (terms[0].y - (terms[0].y - 1) * t)

func smootherstep(t):
	return t * t * t * (t * (t * terms[1].x - terms[1].y) + terms[1].z);

func f(x: float):
	var y := x
	# return 0.35 * pow(x - 0.6, 5) - 1.5 * pow(x - 0.35, 3)
	for term in terms:
		y += pow(x - term.x, floor(term.y)) * term.z
	return y + y_offset
	# return smoothstep(0, 1, x)
	# return smoothstep2(x)
	# return smootherstep(x)

func plot():
	plot_scheduled = true

func _process(_dt):
	# if not plot_scheduled: return
	# DebugTools.print('plot curve')
	plot_scheduled = false
	channel.flush()
	for i in samples:
		var x: float = i / (samples - 1) * x_range - x_range / 2
		channel.write_singles([x, f(x)])
		# channel.write_single(x)
