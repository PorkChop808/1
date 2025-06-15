import tensorrt as trt
print(trt.__version__)
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
print(builder.platform_has_fast_fp16)