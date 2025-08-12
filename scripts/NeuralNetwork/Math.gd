static func normal_distribution(mean: float, sd: float):
	# Box-Muller transform for normal distribution
	var u1 = randf()
	var u2 = randf()
	var z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
	return z0 * sd + mean

static func log2(x: float) -> float:
	const _log2 := log(2.0)
	if x <= 0:
		return 0
	return log(x) / _log2
