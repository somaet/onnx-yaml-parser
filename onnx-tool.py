import onnx_tool
modelpath = 'mxnet_exported_resnet18.onnx'
onnx_tool.model_profile(modelpath) # pass file name