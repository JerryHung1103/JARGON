import json
import tiktoken


def count_tokens(response_text):
    encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    tokens = encoding.encode(response_text)
    return len(tokens)


def format_response(role, response):
    if role == 'attacker':
       return f"<|im_start|>user<|im_sep|>{response}<|im_end|>"
    elif role == 'target':
       return f"<|im_start|>assistant<|im_sep|>{response}<|im_end|>"
    

def remove_think_tags(text: str) -> str:

    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return cleaned_text.strip()

def generate(messages, client, model, **kwargs):
    last_error = None
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "text"},
                **kwargs
            )
            
            if response.choices[0].message.content:
                return remove_think_tags(response.choices[0].message.content)
            else:
                error_msg = f"Error: Empty content in response. Response: {response.choices[0]}"
                print(f"Attempt {attempt+1}/5 failed: {error_msg}")
                last_error = ValueError(error_msg)
                continue
                
        except Exception as e:
            print(f"❌ Attempt {attempt+1}/5 failed: {e}")
            last_error = e
            continue
    

    raise RuntimeError(f"Failed to generate response after 5 attempts. Last error: {last_error}")


import re

def remove_json_markdown(text):
    # Remove JSON code block markers
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Find the position of the first '{'
    first_brace = cleaned.find('{')
    end = cleaned.rfind('}')
    
    # If a '{' is found, keep everything from that position onward
    if first_brace != -1:
        cleaned = cleaned[first_brace:end+1]
    
    return cleaned.strip()

def remove_candidate_lines(text):
    """
    Remove lines containing '\ncandidate{i}' pattern where i can be any number
    
    Args:
        text (str): Input text containing candidate lines
    
    Returns:
        str: Text with candidate lines removed
    """
    # Pattern to match lines containing '\ncandidate{any_number}'
    pattern = r'.*\\candidate\{\d+\}.*\n?'
    
    # Remove lines matching the pattern
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def read_txt_to_string(file_path):
    """
    Read a text file and return its content as a Python string
    
    Args:
        file_path (str): Path to the text file
    
    Returns:
        str: Content of the text file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
