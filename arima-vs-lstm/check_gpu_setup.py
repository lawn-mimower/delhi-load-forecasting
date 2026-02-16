#!/usr/bin/env python3
"""
GPU Setup Diagnostic Tool
Checks if your system is properly configured for GPU acceleration
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode == 0
    except:
        return "", False

def check_nvidia_gpu():
    """Check if NVIDIA GPU is present"""
    print("\n" + "="*60)
    print("1. CHECKING NVIDIA GPU HARDWARE")
    print("="*60)

    # Check for NVIDIA GPU
    output, success = run_command("lspci | grep -i nvidia")
    if success and output:
        print("✅ NVIDIA GPU detected:")
        print(f"   {output}")
    else:
        print("❌ No NVIDIA GPU detected")
        print("   Note: GPU acceleration requires NVIDIA GPU")
        return False

    # Check nvidia-smi
    output, success = run_command("nvidia-smi")
    if success:
        print("\n✅ NVIDIA driver installed")
        # Get driver version
        driver_output, _ = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        if driver_output:
            print(f"   Driver version: {driver_output}")
    else:
        print("\n❌ NVIDIA driver not installed or not working")
        print("   Install with: sudo apt install nvidia-driver-535  # or latest version")
        return False

    return True

def check_cuda():
    """Check CUDA installation"""
    print("\n" + "="*60)
    print("2. CHECKING CUDA INSTALLATION")
    print("="*60)

    # Check nvcc
    output, success = run_command("nvcc --version")
    if success:
        print("✅ CUDA compiler (nvcc) found")
        # Extract version
        lines = output.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"   {line.strip()}")
    else:
        print("❌ CUDA not found")
        print("   Install CUDA from: https://developer.nvidia.com/cuda-downloads")
        return False

    # Check CUDA path
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✅ CUDA_HOME: {cuda_path}")
    else:
        print("⚠️ CUDA_HOME not set")
        print("   Add to ~/.bashrc: export CUDA_HOME=/usr/local/cuda")

    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'cuda' in ld_path.lower():
        print("✅ CUDA libraries in LD_LIBRARY_PATH")
    else:
        print("⚠️ CUDA not in LD_LIBRARY_PATH")
        print("   Add to ~/.bashrc: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")

    return success

def check_cudnn():
    """Check cuDNN installation"""
    print("\n" + "="*60)
    print("3. CHECKING cuDNN INSTALLATION")
    print("="*60)

    # Check for cuDNN library files
    cudnn_paths = [
        "/usr/local/cuda/lib64/libcudnn.so",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so",
        "/usr/local/cuda/include/cudnn.h"
    ]

    cudnn_found = False
    for path in cudnn_paths:
        if os.path.exists(path):
            print(f"✅ Found cuDNN: {path}")
            cudnn_found = True
            break

    if not cudnn_found:
        # Try to find with ldconfig
        output, success = run_command("ldconfig -p | grep cudnn")
        if success and output:
            print("✅ cuDNN libraries found:")
            for line in output.split('\n')[:3]:  # Show first 3 lines
                print(f"   {line.strip()}")
            cudnn_found = True

    if not cudnn_found:
        print("❌ cuDNN not found")
        print("   Download from: https://developer.nvidia.com/cudnn")
        print("   Extract and copy files to /usr/local/cuda/")

    return cudnn_found

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print("\n" + "="*60)
    print("4. CHECKING TENSORFLOW GPU SUPPORT")
    print("="*60)

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")

        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow detects {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"   {gpu}")
        else:
            print("❌ TensorFlow doesn't detect any GPUs")

            # Check if it's built with CUDA
            print("\n   TensorFlow build info:")
            print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
            print(f"   Built with GPU support: {tf.test.is_built_with_gpu_support()}")

            if not tf.test.is_built_with_cuda():
                print("\n   ⚠️ Your TensorFlow is CPU-only version!")
                print("   Reinstall with: pip install tensorflow[and-cuda]")

        return len(gpus) > 0

    except ImportError:
        print("❌ TensorFlow not installed")
        print("   Install with: pip install tensorflow[and-cuda]")
        return False

def check_python_gpu_packages():
    """Check GPU-related Python packages"""
    print("\n" + "="*60)
    print("5. CHECKING PYTHON GPU PACKAGES")
    print("="*60)

    packages = {
        'tensorflow': 'pip install tensorflow[and-cuda]',
        'cupy': 'pip install cupy-cuda12x',  # Adjust version
        'numba': 'pip install numba',
        'torch': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    }

    for package, install_cmd in packages.items():
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            print(f"   Install with: {install_cmd}")

def generate_fix_script():
    """Generate a script to fix common issues"""
    print("\n" + "="*60)
    print("6. FIX SCRIPT")
    print("="*60)

    fix_script = """#!/bin/bash
# GPU Setup Fix Script

# 1. Update system
sudo apt update

# 2. Install NVIDIA driver (Ubuntu/Debian)
sudo apt install nvidia-driver-535  # Or latest version

# 3. Install CUDA 12.1 (adjust version as needed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1

# 4. Set environment variables
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Install Python packages
pip uninstall tensorflow tensorflow-gpu -y
pip install tensorflow[and-cuda]
pip install cupy-cuda12x  # Adjust CUDA version

# 6. Reboot
echo "Please reboot your system after running this script"
"""

    with open('fix_gpu_setup.sh', 'w') as f:
        f.write(fix_script)

    print("📝 Fix script saved to: fix_gpu_setup.sh")
    print("   Run with: bash fix_gpu_setup.sh")

def main():
    print("\n" + "="*60)
    print("🔍 GPU SETUP DIAGNOSTIC TOOL")
    print("="*60)

    all_good = True

    # Run checks
    has_gpu = check_nvidia_gpu()
    all_good = all_good and has_gpu

    if has_gpu:
        has_cuda = check_cuda()
        all_good = all_good and has_cuda

        has_cudnn = check_cudnn()
        all_good = all_good and has_cudnn

    has_tf_gpu = check_tensorflow_gpu()
    all_good = all_good and has_tf_gpu

    check_python_gpu_packages()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if all_good:
        print("✅ Your system is ready for GPU acceleration!")
    else:
        print("❌ GPU setup issues detected")
        print("\nCommon fixes:")
        print("1. Restart Python/Jupyter after installing CUDA")
        print("2. Ensure CUDA version matches TensorFlow requirements")
        print("3. Check TensorFlow-CUDA compatibility:")
        print("   https://www.tensorflow.org/install/source#gpu")

        generate_fix_script()

if __name__ == "__main__":
    main()