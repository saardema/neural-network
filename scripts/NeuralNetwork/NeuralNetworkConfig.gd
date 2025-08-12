class_name NeuralNetworkConfig
extends Resource

@export var learn_rate: float = 0.01
@export var sample_count: int = 100
@export var batch_size: int = 1
@export var iterations: int = 1
@export var wt_std_dev: float = 0.1
@export var problem: String


func save():
	ResourceSaver.save(self , 'res://network_config.tres')
