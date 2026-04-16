import json
import argparse
from tqdm import tqdm
import datetime
import os


parser = argparse.ArgumentParser(description='Process harmful and safe trajectories into training data')
parser.add_argument('--harmful_trajectories_path', type=str, required=True,
                    help='Path to the harmful trajectories JSON file')
parser.add_argument('--safe_trajectories_path', type=str, required=True,
                    help='Path to the safe trajectories JSON file')

args = parser.parse_args()

output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory ensured: ./{output_dir}/")

with open(args.harmful_trajectories_path, 'r', encoding='utf-8') as f:
    harm_data = json.load(f)

with open(args.safe_trajectories_path, 'r', encoding='utf-8') as f:
    safe_data = json.load(f)

num_of_harm_messages = sum(len(trajectories) for trajectories in harm_data.values())
num_of_safe_messages = sum(len(trajectories) for trajectories in safe_data.values())

print(f'Number of harmful messages: {num_of_harm_messages}')
print(f'Number of safe messages: {num_of_safe_messages}')

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_filename = f"training_data_{current_time}.json"
output_path = os.path.join(output_dir, output_filename)

print(f"Processed data will be saved to: {output_path}\n")

from agents.safeguard import filter_harmful_element


for topic, trajectories in tqdm(harm_data.items(), desc="Processing topics"):
    print(f'Processing Query: {topic}')
    
    for sample in tqdm(trajectories, desc=f"  Samples", leave=False):
        # Get the last message (harmful assistant response)
        target_res = sample[-1]['content']
        harmful_element = sample[-1].get('harm', '')

        # Convert harmful response to safe response
        target_safe_res = filter_harmful_element(target_res, harmful_element)

        # Create safe sample and append to safe_data
        safe_sample = sample[:-1] + [{'role': 'assistant', 'content': target_safe_res}]
        
        if topic not in safe_data:
            safe_data[topic] = []
        safe_data[topic].append(safe_sample)

    # Save after each topic (prevents data loss if interrupted)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)


print("Processing completed successfully!")
print(f"Training data saved to: {output_path}")
print(f"Total topics processed: {len(harm_data)}")
