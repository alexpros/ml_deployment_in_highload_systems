#Запускать из контейнера с тритоном
/usr/src/tensorrt/bin/trtexec \
    --onnx=/models/dynamic_batching/1/model.onnx\
    --saveEngine=/models/model.plan \
    --minShapes=input:1x3x224x224\
    --optShapes=input:16x3x224x224\
    --maxShapes=input:32x3x224x224\
    --int8\
    --best\
    --verbose\
    --dumpProfile\
    --percentile=95\
    --exportLayerInfo=/models/trtprofile/LayerInfo.json\
    --exportProfile=/models/trtprofile/profile.json\
    --exportTimes=/models/trtprofile/times.json\
    --separateProfileRun
