@tool
extends Control

const NEURON_LABEL = preload("uid://dl3iq4lf6pmjy")
const Neuron = NeuralNetwork.Neuron

@onready var neuron_labels: Control = %NeuronLabels
@onready var neuron_outline_color: Color = $"../../..".color

@export_range(0, 1) var min_alpha: float = 0.1
@export_range(0, 1) var max_alpha: float = 0.1
@export_range(0, 8) var weight_power: float = 3
@export_range(0, 0.2) var line_width: float = 1:
	set(v):
		line_width = v
		_update_transform()

@export var max_neuron_size: float = 100:
	set(v):
		max_neuron_size = v
		_update_transform()

@export_range(0, 1) var weight_spacing: float = 1:
	set(v):
		weight_spacing = v
		_update_transform()

@export var color_neg: Color
@export var color_pos: Color

@export_range(0, 2) var label_size: float = 1:
	set(v):
		label_size = v
		_update_transform()

@export var show_output_details: bool:
	set(v):
		show_output_details = v
		if nn:
			var render := show_details or show_output_details
			neuron_map[nn.output_layer].container.visible = render
		notify_property_list_changed()

@export var show_details: bool:
	set(v):
		show_details = v
		for layer in neuron_map.values().slice(0, -1):
			layer.container.visible = show_details

		notify_property_list_changed()

@export var show_weights: bool

var selected_neuron: Neuron
var nn: NeuralNetwork
var neuron_radius: float = 40
var p_line_width: float
var neuron_spacing := Vector2()
var font_size: int = 18
var largest_layer_size := 0
var center := size / 2
var font := get_theme_default_font()
var w_spacing: float
var xform := Transform2D()
var transform_pos: Vector2
var transform_scale := Vector2.ONE
var neuron_map: Dictionary[NeuralNetwork.Layer, Dictionary]


func _init():
	resized.connect(_update_transform)
	focus_entered.connect(_on_neuron_selected.bind(null))


func _validate_property(property: Dictionary):
	if property.name == &"show_weights":
		if show_details or show_output_details:
			property.usage &= ~PROPERTY_USAGE_READ_ONLY
		else:
			property.usage |= PROPERTY_USAGE_READ_ONLY


func set_network(network: NeuralNetwork):
	nn = network
	largest_layer_size = 0
	_create_labels()
	_update_transform()


func _create_labels():
	neuron_map = {}

	for wrapper in neuron_labels.get_children():
		for label in wrapper.get_children():
			label.queue_free()
		wrapper.queue_free()

	var layer_idx := -1
	for layer in [nn.input_layer] + nn.layers:
		layer_idx += 1
		largest_layer_size = max(largest_layer_size, layer.size)
		var wrapper := Control.new()
		wrapper.name = 'Layer %d' % layer_idx
		neuron_labels.add_child(wrapper)
		neuron_map[layer] = {'container': wrapper, 'labels': {}}

		for neuron in layer.neurons:
			var neuron_label := NEURON_LABEL.instantiate()
			neuron_label.focus_entered.connect(_on_neuron_selected.bind(neuron))
			neuron_label.focus_exited.connect(_on_neuron_unselected.bind(neuron))
			neuron_label.name = 'Neuron %d' % neuron.neuron_idx
			neuron_label.gui = self
			wrapper.add_child(neuron_label)
			neuron_map[layer]['labels'][neuron] = neuron_label


func _on_neuron_selected(selected: Neuron):
	selected_neuron = selected


func _on_neuron_unselected(unselected: Neuron):
	if selected_neuron == unselected:
		selected_neuron = null


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
	# draw_circle(n_center, neuron_radius, Color(0.1, 0.1, 0.1, 1.0), true, -1, true)
	draw_circle(n_center, neuron_radius * min(1, abs(neuron.activation)), color, true, -1, true)


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
	var layer := nn.layers[layer_idx] if layer_idx >= 0 else nn.input_layer
	var neuron := layer.neurons[neuron_idx]
	var node_pos := get_node_center(layer_idx, neuron_idx)
	var line_height := font_size * 1.25
	var pos := node_pos
	var details_width := 3 * font_size

	pos.x += neuron_radius + 10

	draw_rect(Rect2(pos + Vector2(font_size * 0.3, -line_height), Vector2(details_width * 3, 2.15 * line_height)), Color(0.0, 0.0, 0.0, 0.8))

	# Activation
	draw_string(font, pos, "%5.2f" % neuron.activation, al, details_width, font_size, Color.SEA_GREEN)

	if layer_idx >= 0:
		# Bias
		pos.x += details_width
		draw_string(font, pos, "%5.2f" % neuron.bias, al, details_width, font_size, Color.GOLD)

		# Error
		pos.x += details_width
		draw_string(font, pos, "%5.2f" % neuron.delta, al, details_width, font_size, Color.FIREBRICK)

		# Activation Type
		pos.x -= details_width * 2
		pos.y += line_height * 0.7
		var type: String = Activations.Type.keys()[neuron.activation_type]
		draw_string(font, pos + Vector2(font_size, 0), type, al, 0, font_size * 4 / 5, Color.AZURE)

		if not show_weights: return

		# Weights
		var wfs := maxi(5, font_size * 7 / 7)
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

	draw_set_transform(transform_pos, 0, transform_scale)

	for l in nn.layer_count:
		for n in nn.layers[l].size:
			draw_weights(l, n)

	for l in nn.layer_count:
		for n in nn.layers[l].size:
			draw_neuron(l, n)

	for n in nn.input_layer.size:
		draw_neuron(-1, n)

	var effective_vspace := neuron_spacing.y * transform_scale.y * label_size
	neuron_labels.modulate.a = clamp((effective_vspace - 60) / 3, 0, 1)

	if show_details or show_output_details:
		for neuron in neuron_map[nn.output_layer].labels:
			var label: Control = neuron_map[nn.output_layer].labels[neuron]
			label.update(neuron)

	if show_details:
		for layer in neuron_map.values().slice(0, -1):
			for neuron in layer.labels:
				layer.labels[neuron].update(neuron)


func _gui_input(event: InputEvent) -> void:
	const MASK_SCROLL := (8 | 16 | 32 | 64)
	const MASK_SCROLL_UP_LEFT := (16 | 64)

	if event is InputEventMouseButton and event.pressed:
		if event.button_mask & MASK_SCROLL:
			var direction: float = 1
			if event.button_mask & MASK_SCROLL_UP_LEFT:
				direction = -1
			var logical_before: Vector2 = (event.position - transform_pos) / transform_scale
			transform_scale *= 1 + direction * 0.03
			transform_scale = transform_scale.clampf(1, 6)
			transform_pos = event.position - logical_before * transform_scale
			_update_transform()

	elif event is InputEventMouseMotion:
		if event.button_mask & MOUSE_BUTTON_MASK_LEFT:
			transform_pos += event.relative
			_update_transform()


func _update_transform():
	if not nn: return

	var available_space := Vector2(size.x, min(size.x / nn.layer_count - w_spacing * 5, size.y))
	available_space -= Vector2.ONE * neuron_radius * 2
	neuron_spacing.x = available_space.x / nn.layer_count
	neuron_spacing.y = available_space.y / max(1, (largest_layer_size - 1))
	p_line_width = max(0, line_width * available_space.y * 0.05)
	w_spacing = weight_spacing * neuron_spacing.y / largest_layer_size
	w_spacing = weight_spacing * 40
	neuron_radius = clamp(neuron_spacing.y / 2, 0, max_neuron_size)
	center = size / 2

	transform_pos = transform_pos.clamp(
		- size * transform_scale + size,
		Vector2.ZERO,
	)

	for layer in neuron_map.values():
		for neuron in layer.labels:
			var label: Control = layer.labels[neuron]
			var node_pos := get_node_center(nn.layers.find(neuron.layer), neuron.neuron_idx)
			var vspace: float = (neuron_spacing.y - 8) / label.size.y * transform_scale.y

			label.scale = minf(label_size, vspace) * Vector2.ONE
			label.position = node_pos * transform_scale + transform_pos

			var offset := Vector2(
				neuron_radius * transform_scale.x * 2,
				label.size.y * -0.5 * label.scale.y
			)

			if neuron.layer.is_output:
				label.position.x -= label.size.x * label.scale.x
				offset.x = - offset.x

			label.position += offset

	queue_redraw()
