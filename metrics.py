import json
import numpy as np
import argparse
import scipy.stats as stats

def margin_of_error(x):
    std_dev = np.std(x, ddof=1)
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    return z_score * (std_dev / np.sqrt(len(x)))

def calculate_asr(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    token_counts = []
    total_attacks = len(data)
    successful_attacks = 0
    for attack_data in data.values():
        if attack_data['is_success']:
            successful_attacks += 1
        token_counts.append(attack_data['token_count'])

    error = margin_of_error(token_counts) if len(token_counts) > 1 else 0
    print(f"\nAverage Token Counts {np.mean(token_counts):.3f} ± {error:.3f}")

    if total_attacks > 0:
        asr = (successful_attacks / total_attacks) * 100
    else:
        asr = 0
    
    return asr, total_attacks, successful_attacks

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Attack Statistic from JSON results')
    parser.add_argument('--result_json_file_path', type=str, required=True,
                        help='Path to the JSON file containing attack results')
    args = parser.parse_args()
    
    asr, total_attacks, successful_attacks = calculate_asr(args.result_json_file_path)
    
    print(f"Total attacks: {total_attacks}")
    print(f"Successful attacks: {successful_attacks}")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")