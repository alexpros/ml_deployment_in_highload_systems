/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --minShapes="input:1x3x224x224" \
  --optShapes="input:8x3x224x224" \
  --maxShapes="input:16x3x224x224" \
  --memPoolSize="workspace:2048" \