extends PanelContainer

# signal neuron_focused
# signal neuron_focused

var gui: Control

@onready var label_type: RichTextLabel = %LabelType
@onready var label_activation: RichTextLabel = %LabelActivation
@onready var label_bias: RichTextLabel = %LabelBias
@onready var label_delta: RichTextLabel = %LabelDelta

var sb_focus: StyleBox
# var is_selected: bool:
# 	set(v):
# 		if v != is_selected:
# 			is_selected = v
# 			if is_selected:
# 				add_theme_stylebox_override('panel', sb_focus)
# 			else:
# 				remove_theme_stylebox_override('panel')


func _ready():
	sb_focus = get_theme_stylebox('focus')

	focus_entered.connect(func():
		add_theme_stylebox_override('panel', sb_focus)
		# neuron_focused.emit()
	)

	focus_exited.connect(func():
		remove_theme_stylebox_override('panel')
	)

func update(neuron: NeuralNetwork.Neuron):
	const format := &"%5.2f"

	if neuron is NeuralNetwork.Neuron:
		label_type.text = Activations.Type.keys()[neuron.activation_type]
		label_activation.text = format % neuron.activation
		label_bias.text = format % neuron.bias
		label_delta.text = format % neuron.delta
