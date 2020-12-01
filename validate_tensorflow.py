"""
TensorFlow System/GPU Configuration Validation
Author: Ryan Bales (ryan@balesofdata.com)
"""

import tensorflow as tf

def main():
    print("Testing GPU Confiugration...")
    gpu_devices = tf.config.list_physical_devices('GPU')
    gpu_count = len(gpu_devices)
    
    if gpu_count == 0:
        print("No GPU Devices Found")
    else:
        print(f"{gpu_count} GPUs Found")
        print("\n")
        for gpu in gpu_devices:
            print(f"{gpu.name}")

    print("\n")
    print("Testing TensorFlow Library...")
    tf.debugging.set_log_device_placement(True)

    A = tf.random.normal((5, 5))
    b = tf.random.normal((5, 1))
    print(tf.linalg.solve(A,b))

    print("\n")
    print("TensorFlow Validation Completed")

if __name__ == "__main__":
    main()
