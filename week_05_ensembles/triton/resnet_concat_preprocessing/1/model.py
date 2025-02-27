import os
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:


    def execute(self, requests):
        responses = []
        for request in requests:
            image_1 = pb_utils.get_input_tensor_by_name(request, "IMAGE_V1").as_numpy()
            image_2 = pb_utils.get_input_tensor_by_name(request, "IMAGE_V2").as_numpy()

            preprocessed_images = (image_1 + image_2) / 2
            
            images_tensor = pb_utils.Tensor("IMAGE", np.array(preprocessed_images, dtype=np.float32))
            response = pb_utils.InferenceResponse(output_tensors=(images_tensor,))
            responses.append(response)
        return responses
