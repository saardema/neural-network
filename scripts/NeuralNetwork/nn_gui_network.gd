@tool
extends Control
var nn: NeuralNetwork

var neuron_radius: float = 40
@onready var neuron_outline_color: Color = $"../../..".color
@export_range(0, 1) var min_alpha: float = 0.1
@export_range(0, 1) var max_alpha: float = 0.1
@export_range(0, 8) var weight_power: float = 0.2
@export_range(0, 1) var line_width: float = 1
@export var max_neuron_size: float = 100
@export var weight_spacing_y: float = 1
@export var color_neg: Color
@export var color_pos: Color
@export_range(12, 48) var font_size: int = 18
@export var show_details: bool:
	set(v):
		show_details = v
		notify_property_list_changed()

@export var show_output_details: bool:
	set(v):
		show_output_details = v
		notify_property_list_changed()
@export var show_weights: bool


var p_line_width: float
var neuron_spacing := Vector2()
var largest_layer_size := 0
var center := size / 2
var font := get_theme_default_font()
var w_spacing: float


func _validate_property(property: Dictionary):
	if property.name == &"show_weights":
		if show_details or show_output_details:
			property.usage &= ~PROPERTY_USAGE_READ_ONLY
		else:
			property.usage |= PROPERTY_USAGE_READ_ONLY


func set_network(network: NeuralNetwork):
	nn = network
	largest_layer_size = nn.input_layer.size
	for l in nn.layer_count:
		largest_layer_size = max(largest_layer_size, nn.layers[l].size)


func get_node_center(layer_idx: int, neuron_idx: int) -> Vector2:
	var layer_size := nn.input_layer.size if layer_idx < 0 else nn.layers[layer_idx].size
	var pos := Vector2(
		neuron_radius + (layer_idx + 1) * neuron_spacing.x,
		center.y + (neuron_idx + 0.5 - layer_size / 2.0) * neuron_spacing.y
	)

	return pos


func draw_neuron(layer_idx: int, neuron_idx: int):
	var n_center := get_node_center(layer_idx, neuron_idx)
	var neuron := nn.layers[layer_idx].neurons[neuron_idx] if layer_idx >= 0 else nn.input_layer.neurons[neuron_idx]
	var color := color_pos if neuron.activation > 0 else color_neg
	draw_circle(n_center, neuron_radius + p_line_width, neuron_outline_color, true, -1, true)
	draw_circle(n_center, neuron_radius, Color(0.1, 0.1, 0.1, 1.0), true, -1, true)
	draw_circle(n_center, neuron_radius * min(1, abs(neuron.activation)), color, true)


func draw_weights(layer_idx: int, neuron_idx: int):
	var neuron := nn.layers[layer_idx].neurons[neuron_idx]

	for i in neuron.incoming.size():
		var weight := neuron.weights[i]
		var t := clampf((weight + 1) * 0.5, 0, 1)
		var color := color_neg.lerp(color_pos, t)
		color.a = min_alpha + pow(minf(1, absf(weight)) * (max_alpha - min_alpha), weight_power)

		var start = get_node_center(layer_idx - 1, i)
		var end = get_node_center(layer_idx, neuron_idx)
		var s1 := nn.layers[layer_idx].size
		var s2 := neuron.incoming.size()
		var o1 := neuron_idx + 0.5 - s1 / 2.0
		var o2 := i + 0.5 - s2 / 2.0
		var dir := o1 - o2
		var first := Vector2(start.x, start.y + o1 * w_spacing)
		var last := Vector2(end.x, end.y + o2 * w_spacing)
		var nodes: Array[Vector2]
		nodes.append(first)
		nodes.append(first + Vector2((neuron_spacing.x - neuron_spacing.y * abs(dir)) / 2, 0))
		if dir < 0: nodes.append(Vector2(nodes[-1].x + nodes[-1].y - last.y, last.y))
		elif dir > 0: nodes.append(Vector2(nodes[-1].x + last.y - nodes[-1].y, last.y))
		nodes.append(last)

		draw_polyline(nodes, color, p_line_width, true)

func draw_neuron_text(layer_idx: int, neuron_idx: int):
	const al := HORIZONTAL_ALIGNMENT_RIGHT
	var line_height := font_size * 1.25
	var node_pos := get_node_center(layer_idx, neuron_idx)
	var pos := node_pos

	var layer := nn.layers[layer_idx] if layer_idx >= 0 else nn.input_layer
	var neuron := layer.neurons[neuron_idx]
	var details_width := 3 * font_size

	pos.x += neuron_radius + 10
	pos.y -= line_height * 0.8

	draw_rect(Rect2(pos + Vector2(font_size * 0.3, -line_height), Vector2(details_width * 1.1, 4.15 * line_height)), Color(0.0, 0.0, 0.0, 0.8))

	# Activation
	draw_string(font, pos, "%5.2f" % neuron.activation, al, details_width, font_size, Color.SEA_GREEN)

	if layer_idx >= 0:
		# Bias
		pos.y += line_height
		draw_string(font, pos, "%5.2f" % neuron.bias, al, details_width, font_size, Color.GOLD)

		# Error
		pos.y += line_height
		draw_string(font, pos, "%5.2f" % neuron.delta, al, details_width, font_size, Color.FIREBRICK)

		# Activation Type
		pos.y += line_height * 0.7
		var type: String = Activations.Type.keys()[neuron.activation_type]
		draw_string(font, pos + Vector2(28, 0), type, al, 0, font_size * 2 / 3, Color.AZURE)

		if not show_weights: return

		# Weights
		var wfs := maxi(10, font_size * 7 / 7)
		var wlh := wfs * 1.25
		var columns := ceili((neuron.weights.size()) / 3.0)
		pos = Vector2(node_pos.x - neuron_radius - 20 - wfs * 3, node_pos.y - wlh * 0.8)
		pos.x -= columns * wfs * 3
		pos.y += 3 * wlh
		for w in neuron.weights.size():
			if w % 3 == 0: pos += Vector2(wfs * 3, -3 * wlh)
			draw_string(font, pos, "%5.2f" % neuron.weights[w], al, 3 * wfs, wfs, Color.SILVER)
			pos.y += wlh

func _draw():
	if not nn: return

	var available_space := Vector2(size.x, min(size.x / nn.layer_count - w_spacing * 5, size.y))
	available_space -= Vector2.ONE * (neuron_radius * 2)
	p_line_width = max(0, line_width * available_space.y * 0.05)
	w_spacing = weight_spacing_y * available_space.y * 0.01
	neuron_radius = min(max_neuron_size, w_spacing * 3)
	if show_details or show_output_details: available_space -= Vector2(3 * font_size, 2.5 * font_size)

	center = size / 2
	neuron_spacing.x = available_space.x / nn.layer_count
	neuron_spacing.y = available_space.y / max(1, (largest_layer_size - 1))

	for l in nn.layer_count:
		for n in nn.layers[l].size:
			draw_weights(l, n)

	for l in nn.layer_count:
		for n in nn.layers[l].size:
			draw_neuron(l, n)

	for n in nn.input_layer.size:
		draw_neuron(-1, n)

	if show_details or show_output_details:
		for n in nn.output_layer.size:
			draw_neuron_text(nn.layer_count - 1, n)

	if show_details:
		for l in nn.layer_count - 1:
			for n in nn.layers[l].size:
				draw_neuron_text(l, n)
		for n in nn.input_layer.size:
			draw_neuron_text(-1, n)
