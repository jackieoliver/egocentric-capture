# response_generic_pb2.py/Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc. (http://gopro.com/OpenGoPro).
# Regenerated for protobuf 4.x compatibility

"""Generated protocol buffer code."""

from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16response_generic.proto\x12\nopen_gopro\"@\n\x0fResponseGeneric\x12-\n\x06result\x18\x01 \x02(\x0e\x32\x1d.open_gopro.EnumResultGeneric\"%\n\x05Media\x12\x0e\n\x06\x66older\x18\x01 \x01(\t\x12\x0c\n\x04\x66ile\x18\x02 \x01(\t*\xcf\x01\n\x11\x45numResultGeneric\x12\x12\n\x0eRESULT_UNKNOWN\x10\x00\x12\x12\n\x0eRESULT_SUCCESS\x10\x01\x12\x15\n\x11RESULT_ILL_FORMED\x10\x02\x12\x18\n\x14RESULT_NOT_SUPPORTED\x10\x03\x12!\n\x1dRESULT_ARGUMENT_OUT_OF_BOUNDS\x10\x04\x12\x1b\n\x17RESULT_ARGUMENT_INVALID\x10\x05\x12!\n\x1dRESULT_RESOURCE_NOT_AVAILABLE\x10\x06')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'response_generic_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals['_ENUMRESULTGENERIC']._serialized_start = 144
    _globals['_ENUMRESULTGENERIC']._serialized_end = 351
    _globals['_RESPONSEGENERIC']._serialized_start = 38
    _globals['_RESPONSEGENERIC']._serialized_end = 102
    _globals['_MEDIA']._serialized_start = 104
    _globals['_MEDIA']._serialized_end = 141
