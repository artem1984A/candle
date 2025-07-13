import safetensors
import sys

if len(sys.argv) != 2:
    print("Usage: python debug_tensors.py <path_to_adapter_model.safetensors>")
    sys.exit(1)

path = sys.argv[1]
try:
    with safetensors.safe_open(path, framework="numpy") as f:
        print(f"Tensors in {path}:")
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape}")
except Exception as e:
    print(f"Error: {e}")
