class_name NeuralNetworkGPU
extends NeuralNetworkOld

const MAX_LAYERS := 16

var rd: RenderingDevice
var pipelines: Array[RID]
var shaders: Array[RID]
var uniforms: Array[RID]
var buffers: Array[RID]
var act_buffer: ComputeDoubleBuffer
var buffer_uniform_set: RID

var is_initialized: bool

const shader_names := ['sample', 'forward', 'backward', 'update']
const SHADER_COUNT := 4

func initialize():
	super.initialize()

	if is_initialized: uninitialize()

	rd = RenderingServer.create_local_rendering_device()

	shaders.resize(SHADER_COUNT)
	pipelines.resize(SHADER_COUNT)

	for i in SHADER_COUNT:
		shaders[i] = compile_shader("shaders/%s.glsl" % shader_names[i])
		pipelines[i] = rd.compute_pipeline_create(shaders[i])

	init_buffers()

	is_initialized = true

func uninitialize():
	for rid in pipelines + uniforms + buffers:
		if rid.is_valid(): rd.free_rid(rid)

	for rid in [buffer_uniform_set]:
		if rid.is_valid(): rd.free_rid(rid)

	is_initialized = false

func compile_shader(shader_path: String) -> RID:
	var shader_file := load(shader_path)
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()

	return rd.shader_create_from_spirv(shader_spirv)

func init_buffers():
	var network_config: PackedInt32Array
	network_config.resize(MAX_LAYERS * 3 + 4)
	for l in layers_count:
		network_config[l] = layer_sizes[l]
		network_config[MAX_LAYERS * 1 + l] = weight_offsets[l]
		network_config[MAX_LAYERS * 2 + l] = offsets[l]
	network_config[MAX_LAYERS * 3] = layers_count;
	network_config[MAX_LAYERS * 3 + 1] = neurons_count;

	buffers = []

	var storage_buffer_uniforms: Array[RDUniform] = [
		create_buffer(network_config, 0, false),
		create_buffer(weights, 1),
		create_buffer(biases, 2),
		create_buffer(errors, 3),
		create_buffer(training_data, 4),
		create_buffer(target_outputs, 5),
	]

	buffer_uniform_set = rd.uniform_set_create(storage_buffer_uniforms, shaders[0], 0)
	act_buffer = ComputeDoubleBuffer.new(activations.to_byte_array(), shaders[0], 1, rd)

func create_buffer(data, binding: int, is_storage := true) -> RDUniform:
	var bytes: PackedByteArray = data.to_byte_array()
	var bytes_size := ceilb(bytes.size(), 4) * 4
	var rid: RID
	var uniform := RDUniform.new()

	if is_storage:
		rid = rd.storage_buffer_create(bytes_size, bytes)
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	else:
		rid = rd.uniform_buffer_create(bytes_size, bytes)
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER

	uniform.add_id(rid)
	uniform.binding = binding

	buffers.append(rid)

	return uniform

func ceilb(n: int, base: int) -> int:
	return (n + base - 1) / base

func initialize_weights():
	super.initialize_weights()
	if buffers.size(): network_to_buffers()

func network_to_buffers():
	rd.buffer_update(act_buffer.read_buffer, 0, activations.size() * 4, activations.to_byte_array())
	rd.buffer_update(buffers[1], 0, weights.size() * 4, weights.to_byte_array())
	rd.buffer_update(buffers[2], 0, biases.size() * 4, biases.to_byte_array())
	rd.buffer_update(buffers[3], 0, errors.size() * 4, errors.to_byte_array())

func buffers_to_network():
	activations = rd.buffer_get_data(act_buffer.write_buffer, 0, activations.size() * 4).to_float32_array()
	weights = rd.buffer_get_data(buffers[1], 0, weights.size() * 4).to_float32_array()
	biases = rd.buffer_get_data(buffers[2], 0, biases.size() * 4).to_float32_array()
	errors = rd.buffer_get_data(buffers[3], 0, errors.size() * 4).to_float32_array()

	network_updated.emit()

func train(iterations: int):
	for i in iterations:
		current_sample = (current_sample + 1) % (training_data.size() / inputs_count)
		dispatch()

	buffers_to_network()

func get_push_constant() -> PackedByteArray:
	var pc := var_to_bytes(current_sample).slice(4)
	pc.resize(ceilb(pc.size(), 16) * 16)

	return pc

func dispatch():
	var cl := rd.compute_list_begin()
	var pc := get_push_constant()

	act_buffer.swap()

	rd.compute_list_bind_uniform_set(cl, buffer_uniform_set, 0)
	rd.compute_list_bind_uniform_set(cl, act_buffer.uniform_set, act_buffer.set_idx)

	# Sample
	rd.compute_list_bind_compute_pipeline(cl, pipelines[0])
	rd.compute_list_set_push_constant(cl, pc, pc.size())
	rd.compute_list_dispatch(cl, ceilb(inputs_count, 256), 1, 1)

	# Forward
	rd.compute_list_bind_compute_pipeline(cl, pipelines[1])
	rd.compute_list_set_push_constant(cl, pc, pc.size())
	rd.compute_list_dispatch(cl, ceilb(neurons_count, 256), 1, 1)

	# # Backward
	# rd.compute_list_bind_compute_pipeline(cl, pipelines[2])
	# rd.compute_list_set_push_constant(cl, pc, pc.size())
	# rd.compute_list_dispatch(cl, ceilb(neurons_count, 256), 1, 1)

	# # Update
	# rd.compute_list_bind_compute_pipeline(cl, pipelines[3])
	# rd.compute_list_set_push_constant(cl, pc, pc.size())
	# rd.compute_list_dispatch(cl, ceilb(weights_count, 256), 1, 1)

	rd.compute_list_end()
