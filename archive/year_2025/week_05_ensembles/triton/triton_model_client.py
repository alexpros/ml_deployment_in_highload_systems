import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc

from tritonclient.utils import triton_to_np_dtype


class TritonModelClient:
    """Common client for requesting triton models.
    Provides checks for inputs / outputs parameters.

    Args:
        model_metadata: Model metadata.
        model_config: Model configuration.
        expected_inputs_num [int]: Number of inputs to model, e.g. 1 if only image is expected.
        expected_outputs_num [int]: Number of outputs of model, e.g. 1 if only vector of scores is returned.
        expected_inputs_dtypes [List[Set]]: List of available dtypes for each input, e.g. [{"FP32", "FP16"}].
        expected_outputs_dtypes [List[Set]]: List of available dtypes for each output, e.g. [{"FP32", "FP16"}].
        expected_inputs_dims: List of shapes for each output, e.g. [[3, 224, 224]].
        expected_outputs_dims: List of shapes for each output, e.g. [[1000]].
    """

    def __init__(self,
                 model_metadata, model_config,
                 expected_inputs_num, expected_outputs_num,
                 expected_inputs_dtypes, expected_outputs_dtypes,
                 expected_inputs_dims, expected_outputs_dims):
        self.model_metadata = model_metadata
        self.model_config = model_config
        self._parse_model_config(expected_inputs_num, expected_outputs_num,
                                 expected_inputs_dtypes, expected_outputs_dtypes,
                                 expected_inputs_dims, expected_outputs_dims)

    def _parse_model_config(self,
                            expected_inputs_num, expected_outputs_num,
                            expected_inputs_dtypes, expected_outputs_dtypes,
                            expected_inputs_dims, expected_outputs_dims):
        """Checks the configuration of a model."""
        # Check for number of inputs and outputs
        if len(self.model_metadata.inputs) != expected_inputs_num:
            raise Exception(f"Expecting {expected_inputs_num} inputs in model metadata, "
                            f"got {len(self.model_metadata.inputs)}")
        if len(self.model_config.input) != expected_inputs_num:
            raise Exception(f"Expecting {expected_inputs_num} inputs in model configuration, "
                            f"got {len(self.model_config.input)}")
        if len(self.model_metadata.outputs) != expected_outputs_num:
            raise Exception(f"Expecting {expected_outputs_num} outputs in model_metadate, "
                            f"got {len(self.model_metadata.outputs)}")
        if len(self.model_config.output) != expected_outputs_num:
            raise Exception(f"Expecting {expected_outputs_num} outputs in model configuration, "
                            f"got {len(self.model_config.output)}")

        inputs_names = []
        inputs_shapes = []
        inputs_dtypes = []
        inputs_formats = []
        for i in range(expected_inputs_num):
            input_metadata = self.model_metadata.inputs[i]
            input_config = self.model_config.input[i]

            if input_metadata.datatype not in expected_inputs_dtypes[i]:
                raise TypeError(f"Expecting input datatype to be in {expected_inputs_dtypes[i]}, "
                                f"model's '{self.model_metadata.name}' output '{input_metadata.name}' type "
                                f"is {input_metadata.datatype}.")

            input_batch_dim = (self.model_config.max_batch_size > 0)
            expected_input_ndim = len(expected_inputs_dims[i]) + (1 if input_batch_dim else 0)
            if len(input_metadata.shape) != expected_input_ndim:
                raise Exception(f"Expecting input to have {expected_input_ndim} "
                                f"dimensions, but model '{self.model_metadata.name}' "
                                f"input has {len(input_metadata.shape)}.")

            expected_input_dims_i = expected_inputs_dims[i]
            if self.model_config.max_batch_size > 0:
                expected_input_dims_i = [-1, *expected_inputs_dims[i]]
            if expected_input_dims_i != input_metadata.shape:
                raise Exception(f"Expecting input '{input_metadata.name}' to have shape {expected_input_dims_i}, "
                                f"but model '{self.model_metadata.name}' has {input_metadata.shape}.")

            if isinstance(input_config.format, str):
                format_enum_to_int = dict(mc.ModelInput.Format.items())
                input_config.format = format_enum_to_int[input_config.format]

            inputs_names.append(input_metadata.name)
            inputs_shapes.append(input_metadata.shape)
            inputs_dtypes.append(input_metadata.datatype)
            inputs_formats.append(input_config.format)

        outputs_names = []
        for i in range(expected_outputs_num):
            output_metadata = self.model_metadata.outputs[i]

            if output_metadata.datatype not in expected_outputs_dtypes[i]:
                raise TypeError(f"Expecting output datatype to be in {expected_outputs_dtypes[i]}, "
                                f"model's '{self.model_metadata.name}' output '{output_metadata.name}' type "
                                f"is {output_metadata.datatype}.")

            expected_output_ndim = len(expected_outputs_dims[i]) + (1 if self.model_config.max_batch_size > 0 else 0)
            if len(output_metadata.shape) != expected_output_ndim:
                raise Exception(f"Expecting output to have {expected_output_ndim} "
                                f"dimensions, but model '{self.model_metadata.name}' "
                                f"input has {len(output_metadata.shape)}.")

            expected_output_dims_i = expected_outputs_dims[i]
            if self.model_config.max_batch_size > 0:
                expected_output_dims_i = [-1, *expected_outputs_dims[i]]
            if expected_output_dims_i != output_metadata.shape:
                raise Exception(f"Expecting output '{output_metadata.name}' to have shape {expected_output_dims_i}, "
                                f"but model '{self.model_metadata.name}' has {output_metadata.shape}.")

            outputs_names.append(output_metadata.name)

        self.max_batch_size = self.model_config.max_batch_size
        self.inputs_names = inputs_names
        self.inputs_shapes = inputs_shapes
        self.inputs_dtypes = inputs_dtypes
        self.inputs_formats = inputs_formats
        self.outputs_names = outputs_names

    def request_generator(self, inputs_data):
        """Generates batches in specific tritons request format.

        Args:
            inputs_data [List[np.ndarray]]: List of concatenated inputs for inference.
                Each element in list is data feeded as corresponding model input, e.g. [image_arr, timestamp_arr, ...],
                image_arr with shape [B, 3, 224, 224], timestamp_arr with shape [B, 1].
        """
        client = grpcclient

        if len(inputs_data) != len(self.inputs_names):
            raise Exception(f"Expected {len(self.inputs_names)} inputs, got {len(inputs_data)}.")
        num_inputs = len(inputs_data)

        input_batch_size = len(inputs_data[0])
        max_batch_size = self.max_batch_size if self.max_batch_size > 0 else input_batch_size

        num_batches = int(np.ceil(input_batch_size / max_batch_size))
        for i in range(num_batches):
            inputs = []
            for j in range(num_inputs):
                batch_j = inputs_data[j][i * max_batch_size: (i + 1) * max_batch_size]
                batch_j = np.array(batch_j).astype(triton_to_np_dtype(self.inputs_dtypes[j]))
                inp = client.InferInput(self.inputs_names[j], batch_j.shape, self.inputs_dtypes[j])
                inp.set_data_from_numpy(batch_j)
                inputs.append(inp)
            outputs = [
                client.InferRequestedOutput(name)
                for name in self.outputs_names
            ]

            yield inputs, outputs