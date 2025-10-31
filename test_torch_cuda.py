import torch
import torch.nn as nn
import time

# --- 1. CUDA Device Check and Information ---
def check_cuda_and_get_device():
    """
    Checks for CUDA support, prints device information, and returns the target device.
    """
    print("--- PyTorch and CUDA Version Info ---")
    print(f"PyTorch Version: {torch.__version__}")
    # torch.version.cuda shows the version PyTorch was compiled against
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    print("-----------------------------------")

    if torch.cuda.is_available():
        print("CUDA is available! Running on GPU.")
        print(f"Supported hardware: {torch.cuda.get_arch_list()}")
        device_count = torch.cuda.device_count()
        print(f"Total CUDA Devices Found: {device_count}")
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            # torch.cuda.get_device_capability(i) shows the GPU's compute capability
            capability = torch.cuda.get_device_capability(i)
            print(f"  Device {i}: {name} (Compute Capability: {capability[0]}.{capability[1]})")
            
        # Select the first available GPU (device 0)
        device = torch.device("cuda:0")
    else:
        print("CUDA is NOT available. Falling back to CPU.")
        device = torch.device("cpu")
        
    return device

# --- 2. Define a Simple Neural Network ---
class SimpleNet(nn.Module):
    """
    A simple Feed-Forward Neural Network with two linear layers.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# --- 3. Run the Network on the Selected Device ---
def run_network(device):
    """
    Instantiates the model, moves it and data to the device, and runs a forward pass.
    """
    print("-" * 40)
    print(f"Target Device for Model/Data: {device}")
    
    # Hyperparameters
    INPUT_SIZE = 10
    HIDDEN_SIZE = 50
    OUTPUT_SIZE = 2
    BATCH_SIZE = 64
    
    # Instantiate the model and move it to the device
    model = SimpleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    print(f"Model successfully moved to: {next(model.parameters()).device}")

    # Create dummy input data and move it to the device
    dummy_input = torch.randn(BATCH_SIZE, INPUT_SIZE).to(device)
    print(f"Dummy input data successfully moved to: {dummy_input.device}")
    
    # Measure simple forward pass time (a very basic benchmark)
    start_time = time.time()
    
    # Run the forward pass
    output = model(dummy_input)
    
    end_time = time.time()
    
    # Ensure the operation is complete before measuring time, especially for CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    print("-" * 40)
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Time taken for forward pass: {end_time - start_time:.4f} seconds (Note: This is a very rough measure).")
    print("-" * 40)

# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Check CUDA and get the best available device
    selected_device = check_cuda_and_get_device()
    
    # 2. Run the network on the selected device
    run_network(selected_device)

    print("Program finished executing.")