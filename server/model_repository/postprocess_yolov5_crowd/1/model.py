import numpy as np
import sys
import json
import io
# from PIL import Image
import os
import triton_python_backend_utils as pb_utils
import torch
from yolov5_postprocess import non_max_suppression

class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
        
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            output_model = in_0.as_numpy()
            output_model = torch.from_numpy(output_model)
            boxes =  np.array(non_max_suppression(output_model))
            out_tensor_0 = pb_utils.Tensor("OUTPUT_0",
                                           boxes.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')