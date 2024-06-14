import numpy as np
import os

INPUT_SHAPE = (1, 4, 128, 1200)
DATA_DIR = 'data/train'  # Replace with your directory path


def load_data():
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith('.npy'):
            file_path = os.path.join(DATA_DIR, file_name)
            data = np.load(file_path).reshape(1, 5, 128, 1200).astype(np.float32) 
            data = data[:, :4, :, :]  # Select only the required shape
            yield {
                "input": data
            }


# Example usage
if __name__ == "__main__":
    for batch in load_data():
        print(batch["input"].shape)
