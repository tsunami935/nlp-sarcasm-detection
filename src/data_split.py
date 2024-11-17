import json
import random

def load_data(file_path):
    """Load data from the JSON file."""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def split_data(data):
    """Split the data into 70%, 20%, 10% with balanced is_sarcastic values."""
    sarcastic = [item for item in data if item['is_sarcastic'] == 1]
    non_sarcastic = [item for item in data if item['is_sarcastic'] == 0]

    # Shuffle the lists to ensure random distribution
    random.shuffle(sarcastic)
    random.shuffle(non_sarcastic)

    # Calculate the number of items for each split
    total_sarcastic = len(sarcastic)
    total_non_sarcastic = len(non_sarcastic)
    
    # Determine split sizes
    train_size_sarcastic = int(0.7 * total_sarcastic)
    val_size_sarcastic = int(0.2 * total_sarcastic)

    train_size_non_sarcastic = int(0.7 * total_non_sarcastic)
    val_size_non_sarcastic = int(0.2 * total_non_sarcastic)

    # Create the splits for sarcastic and non-sarcastic
    train_data = sarcastic[:train_size_sarcastic] + non_sarcastic[:train_size_non_sarcastic]
    val_data = sarcastic[train_size_sarcastic:train_size_sarcastic+val_size_sarcastic] + non_sarcastic[train_size_non_sarcastic:train_size_non_sarcastic+val_size_non_sarcastic]
    test_data = sarcastic[train_size_sarcastic+val_size_sarcastic:] + non_sarcastic[train_size_non_sarcastic+val_size_non_sarcastic:]

    # Shuffle the splits to ensure random distribution within each set
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data

def save_data(split_data, filenames):
    """Save the split data into separate files."""
    for data, filename in zip(split_data, filenames):
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

def main():
    # Specify the input and output file paths
    input_file = 'Sarcasm_Headlines_Dataset.json'  
    output_files = ['train.json', 'val.json', 'test.json']

    # Step 1: Load the data from the input file
    data = load_data(input_file)

    # Step 2: Split the data
    train_data, val_data, test_data = split_data(data)

    # Step 3: Save the splits into separate files
    save_data([train_data, val_data, test_data], output_files)

    print(f"Data successfully split into {output_files[0]}, {output_files[1]}, and {output_files[2]}.")

if __name__ == '__main__':
    main()
