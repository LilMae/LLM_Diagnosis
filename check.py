import torch
print("torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Supported architectures:", torch.cuda.get_arch_list())
print("Device capability:", torch.cuda.get_device_capability(0))
print("Device name:", torch.cuda.get_device_name(0))
print("Device count:", torch.cuda.device_count())