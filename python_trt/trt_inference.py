import os
import sys
import time
import ctypes
import argparse
import glob
import numpy as np
import tensorrt as trt
from PIL import Image
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms

class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        print("Loading engine: ", engine_path)
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        print("=========================================")
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0
        print("Init model successfully!!!")
        print("=========================================")

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        cuda.memcpy_htod(self.inputs[0]['allocation'], batch)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]

    def process(self, batch):
        # Run inference
        outputs = self.infer(batch)
        return outputs

def main(args):
    executor = TensorRTInfer(args.trt_engine)

    print("Done")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
    parser.add_argument('--trt_engine', type=str, 
                        default='D:/projects/TritonDeployment/server/model_repository/clothes_segmentation/1/model_densenet_bs_dynamic.engine', 
                        help='Feature extractor')

    args = parser.parse_args()

    main(args)

