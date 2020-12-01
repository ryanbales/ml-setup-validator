"""
PyTorch System/GPU Configuration Validation
Author: Ryan Bales (ryan@balesofdata.com)
"""
import torch

def main():
    print("Testing GPU Confiugration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        print("No GPU Devices Found")
    else:
        print(f"{torch.cuda.device_count()} GPUs Found")
        print("Primary GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("VRAM: ", round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), "GB")
  
    print("\n")
    print("Testing PyTorch Library...")

    A = torch.rand(5, 5).to(device)
    b = torch.rand(5, 1).to(device)
    print(torch.solve(b, A))

    print("\n")
    print("PyTorch Validation Completed")

if __name__ == "__main__":
    main()
