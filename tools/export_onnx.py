import shutil
import torch

try:
    import onnx
    import onnx.utils
except ImportError:
    onnx = None
try:
    import onnxsim
except ImportError:
    onnxsim = None

def export(model, outfile, verbose=False, batch=1, input_w=256, input_h=256, dynamic=False):
    dummy_input = torch.randn(batch, 3, input_h, input_w)
    input_names=['input']
    output_names=['output']
    if dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    else:
        dynamic_axes = None
    torch.onnx.export(
        model, dummy_input, outfile, verbose=verbose,
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names)

def simplify(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unsimplified.onnx')
        shutil.copyfile(outfile, infile)

    simplified_model, check_ok = onnxsim.simplify(infile, check_n=3, perform_optimization=False)
    assert check_ok
    onnx.save(simplified_model, outfile)