import os
import sys
import time

import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from triton_model_client import TritonModelClient

input_path = "./infer_inputs.npy"
output_path = "./infer_outputs.npy"

URL = "localhost:19094"
VERBOSE = False

MODEL_NAME = "resnet_inference"
MODEL_VERSION = ""

EXPECTED_INPUTS_NUM = 1
EXPECTED_OUTPUTS_NUM = 1
EXPECTED_INPUTS_DTYPES = [{"FP32"}]
EXPECTED_INPUTS_DIMS = [[ 3, 224, 224 ]]
EXPECTED_OUTPUTS_DIMS = [[1000]]
EXPECTED_OUTPUTS_DTYPES = [{"FP32"}]

def main():
    # Create gRPC client for communication with the server
    triton_client = grpcclient.InferenceServerClient(url=URL, verbose=VERBOSE)

    # Get model metadata
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION
        )

    except InferenceServerException as e:
        print(f"Failed to retrieve the metadata: {e}.")
        sys.exit(1)
    # Get model configuration
    try:
        model_config = triton_client.get_model_config(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )
        model_config = model_config.config
    except InferenceServerException as e:
        print(f"Failed to retrieve the config: {e}.")
    client = TritonModelClient(model_metadata, model_config,
                               EXPECTED_INPUTS_NUM, EXPECTED_OUTPUTS_NUM,
                               EXPECTED_INPUTS_DTYPES, EXPECTED_OUTPUTS_DTYPES,
                               EXPECTED_INPUTS_DIMS, EXPECTED_OUTPUTS_DIMS)

    inputs_data = [np.load(input_path).astype(np.float32)]
    
    print(f"Input shape {inputs_data[0].shape}, {len(inputs_data)}")

    # END OF CUSTOM PART

    print("INPUT:")
    for i in range(len(inputs_data)):
        print("\t", client.inputs_names[i], inputs_data[i].shape)

    # Send requests
    print("INFERENCE:")
    start_time = time.time()
    num_batches = 0
    try:
        all_outputs = {}
        for num, (inputs, outputs) in enumerate(client.request_generator(inputs_data)):
            start = time.time()
            response = triton_client.infer(MODEL_NAME,
                                           inputs,
                                           model_version=MODEL_VERSION,
                                           outputs=outputs)
            for output_name in client.outputs_names:
                if output_name not in all_outputs:
                    all_outputs[output_name] = []
                all_outputs[output_name].append(response.as_numpy(output_name))
            num_batches += 1

    except InferenceServerException as e:
        print(f"Inference failed: {e}.")
        sys.exit(1)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Processed {num_batches} batches")
    print(f"Total time: {total_time:.3f}s, avg time per batch: {total_time / num_batches:.3f}s")
    # CUSTOM PART: saving results
    # images = all_outputs["embedding"]
    for key in all_outputs.keys():
        print(key)
        all_outputs[key] = np.concatenate(all_outputs[key], axis=0)
        print()

    np.save(output_path, all_outputs["embedding"])

if __name__ == "__main__":
    main()
