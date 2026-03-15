import triton_python_backend_utils as pb_utils
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from vllm import LLM, SamplingParams
import torch
import numpy as np
import json
import os
import gc


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have 'TritonPythonModel' as the class name.
    """

    def initialize(self, args):
        """Called once when the model is loaded. Used to initialize the model.
        
        Parameters
        ----------
        args : dict
            A dictionary containing model configurations.
        """
        # Load the LLM configuration parameters from the JSON file
        with open(os.path.join(args["model_repository"], "model.json"), 'r', encoding='utf-8') as config_file:
            llm_config = json.load(config_file)
            sampling_params_config = llm_config.pop("sampling_params", {})

        # Initialize the LLM model from VLLM
        weights_path = os.path.join(args["model_repository"], args["model_version"], "weights")
        self.llm_model = LLM(
            model=weights_path,
            tokenizer=weights_path,
            **llm_config
        )

        # Sampling parameters for text generation
        self.sampling_params = SamplingParams(**sampling_params_config)

    def decode(self, strings):
        """Helper function to decode byte strings to UTF-8."""
        return [s.decode("utf-8") for s in strings.squeeze(1)]

    def execute(self, requests):
        """Handles inference requests.
        
        Parameters
        ----------
        requests : list
            A list of pb_utils.InferenceRequest

        Returns
        -------
        list
            A list of pb_utils.InferenceResponse. The length of this list must
            match the length of the `requests` list.
        """
        responses = []

        for request in requests:
            # Decode input tensor
            inputs = self.decode(pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy())
            
            # Generate outputs using the LLM model
            llm_outputs = self.llm_model.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False)

            # Extract text outputs
            text_outputs = [output.outputs[0].text for output in llm_outputs]
            
            # Create output tensor and response
            output_tensor = pb_utils.Tensor("text_output", np.array(text_outputs, dtype=np.object_).reshape(-1, 1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """Called once when the model is unloaded. Used for cleanup."""
        destroy_model_parallel()
        destroy_distributed_environment()
        if hasattr(self, 'llm_model'):
            del self.llm_model.llm_engine.model_executor.driver_worker
            del self.llm_model  # Free up memory by deleting the model instance
        gc.collect()
        torch.cuda.empty_cache()

