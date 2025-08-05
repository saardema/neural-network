class_name Activations
extends RefCounted

enum Type {Pass, TanH, Sigmoid, ReLu, SoftMax, LeakyReLu, PreLu}

func get_function(type: Type) -> ActivationFunction:
	return function_map.get(type, function_map.get(Type.Sigmoid))

var function_map: Dictionary = {
	Type.TanH: Tanh.new(),
	Type.Sigmoid: Sigmoid.new(),
	Type.ReLu: ReLu.new(),
	Type.SoftMax: SoftMax.new(),
	Type.PreLu: PreLu.new(),
	Type.Pass: Pass.new()
}

@abstract class ActivationFunction:
	@abstract func activate(x: float) -> float
	@abstract func differentiate(a: float) -> float

class Pass extends ActivationFunction:
	func activate(x: float) -> float:
		return x

	func differentiate(_a: float) -> float:
		return 1.0

class PreLu extends ActivationFunction:
	var prelu_alpha: float = 0.25

	func activate(x: float) -> float:
		return max(prelu_alpha * x, x)

	func differentiate(a: float) -> float:
		return 1.0 if a > 0 else prelu_alpha

class SoftMax extends ActivationFunction:
	func activate(x: float) -> float:
		return exp(x) / (1 + exp(x))

	func differentiate(a: float) -> float:
		return a * (1 - a)

class ReLu extends ActivationFunction:
	func activate(x: float) -> float:
		return max(0, x)

	func differentiate(_a: float) -> float:
		return 1 if _a > 0 else 0

class Sigmoid extends ActivationFunction:
	func activate(x: float) -> float:
		return 1.0 / (1.0 + exp(-x))

	func differentiate(a: float) -> float:
		return a * (1.0 - a)

class Tanh extends ActivationFunction:
	func activate(x: float) -> float:
		var p := exp(x)
		var n := exp(-x)
		return (p - n) / (p + n)

	func differentiate(a: float) -> float:
		return 1.0 - a * a
