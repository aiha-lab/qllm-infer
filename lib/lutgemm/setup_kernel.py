# This code is for testing kernel-level performance
# Change the compute capability appropriately (e.g., A100: 80, A6000: 86)

import os
import subprocess

def run_command(command, cwd=None):
    """Run a shell command and print its output."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True, cwd=cwd)

def main():
    repo_url = "https://github.com/naver-aics/lut-gemm.git"
    repo_name = "lut-gemm"
    build_dir = "build"
    
    # Clone the repository
    if not os.path.exists(repo_name):
        run_command(f"git clone {repo_url}")
    else:
        print(f"Repository '{repo_name}' already exists. Skipping clone.")
    
    # Navigate into the repository
    os.chdir(repo_name)
    
    # Create and navigate to the build directory
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(build_dir)
    
    # Run cmake and make
    run_command("cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..")
    run_command("make -j8")
    
    # Run tests
    run_command("./tests/tests")

    print("lut-gemm tests completed successfully!")

if __name__ == "__main__":
    main()
