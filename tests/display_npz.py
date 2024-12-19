import numpy as np

def display_npz_contents(file_path):
    """
    Load and display the contents of a .npz file.
    
    Parameters:
    file_path (str): Path to the .npz file
    """
    # Load the npz file
    with np.load(file_path) as data:
        # Get list of all arrays in the file
        array_names = data.files
        print(f"Arrays in {file_path}:")
        print("-" * 40)
        
        # Loop through and display info about each array
        for name in array_names:
            array = data[name]
            print(f"\nArray name: {name}")
            print(f"Shape: {array.shape}")
            print(f"Data type: {array.dtype}")
            if name == "predictions":
                large_preds = np.sum(np.abs(array) > 0.01)
                print(f"Number of predictions > 0.01: {large_preds} (out of {array.size})")
            if name == "confidences":
                large_confs = np.sum(np.abs(array) > 0.01)
                print(f"Number of confidences > 0.01: {large_confs} (out of {array.size})")
            if name != "dates" and name != "test_loss":
                print(f"First few elements: {array[:20, :20]}")  # Show first elements
            elif name == "dates":
                print(f"First few elements: {array[:20]}")  # Show first elements
            # elif name == "test_loss":
            #     print(f"First few elements: {array}")
            
            # For small arrays, optionally show full contents
            if array.size < 20:  # Adjust this threshold as needed
                print("Full contents:")
                print(array)

# Example usage
file_path = "data/evaluation_results.npz"
display_npz_contents(file_path)