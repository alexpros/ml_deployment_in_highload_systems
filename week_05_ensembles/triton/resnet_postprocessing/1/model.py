import os
import json
import numpy as np
import triton_python_backend_utils as pb_utils
import shutil

class TritonPythonModel:

    def execute(self, requests):
        responses = []
        for request in requests:
            
            embeds = pb_utils.get_input_tensor_by_name(request, "embedding").as_numpy()
            score = embeds.sum()
            
            score_tensor = pb_utils.Tensor("score", np.array(score, dtype=np.float32))
            response = pb_utils.InferenceResponse(output_tensors=(score_tensor,))
            responses.append(response)
        return responses
