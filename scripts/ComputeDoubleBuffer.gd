class_name ComputeDoubleBuffer

var _state: int
var _rd: RenderingDevice
var _shader: RID
var _uniforms: Array[RDUniform] = [RDUniform.new(), RDUniform.new()]
var _buffers: Array[RID] = [RID(), RID()]
var set_idx: int
var uniform_set: RID
var read_buffer: RID
var write_buffer: RID

func _init(data: PackedByteArray, shader: RID, set_index: int, rd: RenderingDevice):
	_rd = rd
	_shader = shader
	set_idx = set_index

	for i in 2:
		_buffers[i] = _rd.storage_buffer_create(data.size(), data)
		_uniforms[i].uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		_uniforms[i].add_id(_buffers[i])

	swap()

func swap():
	read_buffer = _buffers[_state]
	write_buffer = _buffers[1 - _state]
	_uniforms[0].binding = _state
	_uniforms[1].binding = 1 - _state
	if uniform_set.is_valid(): _rd.free_rid(uniform_set)
	uniform_set = _rd.uniform_set_create(_uniforms, _shader, set_idx)
	_state = !_state