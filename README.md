# Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#attack-results">View Results</a> •
  <a href="#processing-trajectories-for-training">Data Pipeline</a>
</p>

---

## Overview

This repository contains the implementation of **JARGON**, a jailbreak attack that leverages domain‑specific contexts to bypass LLM safety mechanisms. By embedding harmful requests within legitimate academic, technical domains, JARGON exposes significant vulnerabilities in current alignment techniques. The framework supports multi‑round conversational attacks, prompt optimization, belief state tracking, and automatic extraction of harmful knowledge.
 
> **Authors**: Ki Sen Hung, Xi Yang, Chang Liu, Haoran Li, Kejiang Chen, Changxuan Fan, Tsun On Kwok, Weiming Zhang, Xiaomeng Li, Yangqiu Song

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JerryHung1103/JARGON.git
cd JARGON
```


### 2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate JARGON
```

### 3. Configure API keys and endpoints 
Create a .env file in the project root (or copy from .env.example) with the following variables:

```bash
HUGGINGFACE_HUB_TOKEN=your_hf_token_here

# Attacker model
ATTACKER_API_KEY=your_api_key_here
ATTACKER_BASE_URL=your_base_url_here

# Optimizer model
OPTIMIZER_API_KEY=your_api_key_here
OPTIMIZER_BASE_URL=your_base_url_here

# Judge model
JUDGE_API_KEY=your_api_key_here
JUDGE_BASE_URL=your_base_url_here

# Summary helper model
SUMMARY_API_KEY=your_api_key_here
SUMMARY_BASE_URL=your_base_url_here

# Refusal checker model
REFUSAL_CHECKER_API_KEY=your_api_key_here
REFUSAL_CHECKER_BASE_URL=your_base_url_here

# Target model
TARGET_API_KEY=your_api_key_here
TARGET_BASE_URL=your_base_url_here

# Knowledge extractor (for purification)
KNOWLEDGE_EXTRACTOR_API_KEY=your_api_key_here
KNOWLEDGE_EXTRACTOR_BASE_URL=your_base_url_here

# Safeguard (optional, only if enable_safeguard=True)
SAFEGUARD_API_KEY=your_api_key_here
SAFEGUARD_BASE_URL=your_base_url_here
```

>💡Note: All BASE_URL values should point to the API endpoint of your chosen LLM provider (e.g., OpenAI‑compatible endpoint). You can use the same API key for multiple roles if your provider allows.

### 4. Configure `config/config.yml`
The main settings are controlled through `config.yml`. Below are some example and explanations:

```yaml
jailbreak_setting:
  mode: 'Attack' 
  # Mode can be 'Attack' or 'Distillation'
  # - 'Attack': Uses LLM judge to determine successful jailbreaks
  # - 'Distillation': No judge is used. Attack ends when max_rounds_per_attack is reached.
  # NOTE: Distillation mode cannot be used with early_stop enabled.

  max_retries: 3
  # Number of full retry attempts per test case (outer loop)

  max_trials_per_retry: 2
  # Number of trials (conversations) within each retry

  max_rounds_per_attack: 4
  # Maximum number of attack rounds (turns) in one conversation/trial

  example_injection_threshold: 2
  # After how many retries will successful jailbreak examples be injected into the attacker prompt.
  # Set to a low number to enable few-shot learning earlier.

  optimization_variants: 8
  # Number of prompt variants generated during each optimization step.
  # Higher = more diversity, but higher API cost.

  diverse_attack_ratio: 0.5
  # Ratio of variants that use diversity/role-playing techniques (0.0 ~ 1.0)
  # 0.5 means 50% diverse variants, 50% standard variants.

  early_stop: False
  # If True, stops the attack immediately once a successful jailbreak is found.
  # Greatly saves time and API cost in 'Attack' mode.
  # Cannot be used together with 'Distillation' mode.

  record_trajectories: False
  # If True, saves full attack trajectories (both harmful and safe) into separate folders.
  # Useful for analysis or collecting data for fine-tuning refusal models.

  num_workers: 5
  # Number of parallel worker processes.
  # Adjust according to your CPU cores and API rate limits.

  enable_safeguard: False
  # Enables dynamic safeguard simulation on the target model (for testing robustness against defenses).

# ...
target_model:                                  
  model: "deepseek-chat"    
  base_url: "${TARGET_BASE_URL}"   
  api_key: "${TARGET_API_KEY}"                              
  generate_kwargs:
    temperature: 0.0

attacker:           
  model: "deepseek-chat"                           
  base_url: "${ATTACKER_BASE_URL}" 
  api_key: "${ATTACKER_API_KEY}"          
  generate_kwargs:
    temperature: 0.3
# ... other models follow the same pattern
```


## Usage

### 1. Paper Content Extraction (Optional)

To prepare a new research paper (e.g., as context for the Jargon attack), use the `paper_content_extractor.py` script. It processes a PDF file, extracts the core academic content (removing PDF artifacts, headers, footers, etc.), and generates a structured JSON summary containing the title, abstract, methodology, and full cleaned text.

```bash
python paper_content_extractor.py --pdf_file_path path_to_paper.pdf
```

### 2. Execute Attacks

Run the attacks:

```bash
python main.py
```
<p>When you run an attack, the following components execute in a coordinated loop across retries, trials, and conversation rounds:</p>

<table>
    <thead>
        <tr>
            <th>Component</th>
            <th>Role</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Jailbreak Engine</strong> (<code>jailbreak_engine.py</code>)</td>
            <td>
                <ul>
                    <li>Orchestrates the entire attack: retries, trials, and multi‑round conversations.</li>
                    <li>Injects successful examples from cache after <code>example_injection_threshold</code> retries (few‑shot prompting).</li>
                    <li>Calls the attacker to generate a plan and the next query, then updates belief states.</li>
                    <li>Optionally triggers the <strong>Prompt Optimizer</strong> when <code>attackFlag == 1</code>.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Attacker</strong></td>
            <td>
                <ul>
                    <li>Generates the next user query (<code>nextPrompt</code>), a plan (<code>suggestedTactics</code>), and an <code>attackFlag</code>.</li>
                    <li>Updates belief states based on conversation history.</li>
                    <li>Produces criticism and prompt notes after failed trials.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Prompt Optimizer</strong> (<code>agents/optimizer.py</code>)</td>
            <td>
                <ul>
                    <li>Creates multiple variants of the current query:
                        <ul>
                            <li><strong>Paraphrase variants</strong> (human‑like rewrites).</li>
                            <li><strong>Diverse scenario attack variants</strong> (domain camouflage, e.g., film scripts, game design).</li>
                        </ul>
                    </li>
                    <li>For each variant, sends it to the target model and evaluates the response.</li>
                    <li>Selects the variant with the highest evaluator score (preferring non‑refusal responses).</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Target Model</strong></td>
            <td>
                <ul>
                    <li>The LLM under attack. Responds to each user message (attacker query).</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Evaluator</strong> (<code>agents/evaluator.py</code>)</td>
            <td>
                <ul>
                    <li>Scores each target response on a <strong>1–5 scale</strong> (1 = no progress, 5 = full jailbreak).</li>
                    <li>Uses a strict rubric based on the attack goal.</li>
                    <li>Performs multiple inferences (default 3) and takes the majority score/reason.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Knowledge Extractor</strong> (<code>knowledge_extractor_generate</code>)</td>
            <td>
                <ul>
                    <li>Extracts concise harmful claims from verbose target responses.</li>
                    <li>Wraps extracted knowledge in <code>&lt;harm&gt;</code> tags and accumulates it across rounds.</li>
                    <li>Caches successful extractions in <code>harmful_trajectory_cache_file</code>.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Refusal Checker</strong> (<code>agents/refusal_checker_utils.py</code>)</td>
            <td>
                <ul>
                    <li>Detects whether the target model explicitly refused the request.</li>
                    <li>Uses majority voting over 3 inferences.</li>
                    <li>Helps the optimizer avoid selecting refused responses.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Judge</strong> (optional, injected via <code>judge</code> parameter)</td>
            <td>
                <ul>
                    <li>Performs final harmful content detection on the accumulated <code>&lt;harm&gt;</code> text.</li>
                    <li>Returns a binary success flag and a rationale.</li>
                    <li>If <code>early_stop</code> is enabled, a successful judge stops the attack immediately.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Safeguard</strong> (<code>agents/safeguard.py</code>) – optional (enabled via <code>enable_safeguard</code>)</td>
            <td>
                <ul>
                    <li>Analyzes the current and previous queries to detect jailbreak patterns.</li>
                    <li>Dynamically generates a safety system prompt for the target model.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td><strong>Results &amp; Caching</strong></td>
            <td>
                <ul>
                    <li>Final attack outcome is written to <code>out_dir</code> of config.yml.</li>
                    <li>Successful conversation transcripts are cached in <code>successfulExampleCachePath</code> for future few‑shot injection.</li>
                    <li>Harmful and safe trajectories are saved to separate cache files (enabled via <code>record_trajectories</code> in config.yml): <code>harmful_trajectory_cache_file</code>, <code>safe_trajectory_cache_file</code>.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>



## Attack Results
After running the attack, You can calculate key metrics using the built-in script:
```
python metrics.py --result_json_file_path your_results.json
```

## Processing Trajectories for Training 

When `record_trajectories: True`, the system saves harmful and safe trajectories separately.
Use `process_data.py` to clean harmful trajectories and generate training data for fine-tuning.

```
python process_data.py \
  --harmful_trajectories_path  your_harmful_trajectories_.json \
  --safe_trajectories_path your_safe_trajectories.json
```

It removes and refine the harmful elements from the last assistant response in harmful trajectories
Merges the cleaned responses into the safe data then saves the result to `./training_data/`

