# save as preprocess_mydata.py
from datasets import load_from_disk
import argparse, os, re

def parse_args():
    parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to veRL format")
    parser.add_argument("--dataset_path", required=True, help="Path to the HuggingFace dataset")
    parser.add_argument("--out_dir", required=True, help="Output directory for the converted dataset")
    return parser.parse_args()

def extract_dataset_name(dataset_path):
    """
    Extract the dataset name from the dataset path.
    If the path is a file path, extract the filename without extension.
    If the path is a HuggingFace dataset ID, use the last part of the ID.
    """
    # Check if it's a file path
    if os.path.exists(dataset_path):
        return os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Otherwise, assume it's a HuggingFace dataset ID (e.g., "myorg/mydata")
    return dataset_path.split('/')[-1]

def main():
    args = parse_args()
    
    # Extract dataset name from the path
    dataset_name = extract_dataset_name(args.dataset_path)
    
    # Load the dataset
    ds = load_from_disk(args.dataset_path)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process and save the dataset splits

    output_path = f"{args.out_dir}.parquet"

    ds.map(
        make_map_fn(args), 
        with_indices=True
    ).to_parquet(output_path)
    
    print(f"Dataset converted and saved to {output_path}")


# 1️⃣ optional helper – extract your “ground truth”
def extract_gt(example):
    return example["label"]            # adapt to your dataset

# 2️⃣ build the mapping function veRL asks for
def make_map_fn(args):
    DATA_SOURCE = extract_dataset_name(args.dataset_path)
    def _map(example, idx):
        prompt_txt = example["prompt"].strip()
        # if your template needs a system msg, add it here
        row = {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": prompt_txt}],
            "ability": "open_ended",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['verification_info']
            },
            "extra_info": {"gold_standard_solution": example['gold_standard_solution']}
        }
        return row
    return _map

if __name__ == "__main__":
    main()
