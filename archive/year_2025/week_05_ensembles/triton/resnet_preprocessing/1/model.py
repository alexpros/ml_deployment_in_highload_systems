import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import hashlib
from utils import preproc_images
import shutil

class TritonPythonModel:

    def initialize(self, args):
        model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []
        for request in requests:
            image_filenames = [
                text[0].decode("utf-8")  for text in pb_utils.get_input_tensor_by_name(request, "IMAGE_PATH").as_numpy()
            ]

            preprocessed_images = preproc_images(image_filenames, target_size=224)
            
            images_tensor = pb_utils.Tensor("IMAGE_V1", np.array(preprocessed_images, dtype=np.float32))
            response = pb_utils.InferenceResponse(output_tensors=(images_tensor,))
            responses.append(response)
        return responses
