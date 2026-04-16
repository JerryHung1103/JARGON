import PyPDF2
import re
from typing import Dict
import os
import argparse
import openai
import json

from utils import generate,remove_json_markdown
from prompts import get_paper_summary_prompt, paper_output_format_instruction
from config.config_loader import ConfigLoader
config_loader = ConfigLoader()
from pathlib import Path


summary_model=config_loader.summary_model['model']
summerizer_generate = lambda messages: generate(
    messages, 
    client=openai.OpenAI(
        base_url=config_loader.summary_model['base_url'],
        api_key=config_loader.summary_model['api_key']
    ),
    model=summary_model, 
    **config_loader.summary_model['generate_kwargs']
)



def get_clean_pdf_prompt(pdf_content):
    return f'''**Role and Goal:**
You are an expert Text Cleaning Specialist and Academic Content Extraction Engine. Your primary goal is to accurately identify and extract the **complete, pure academic body** of the research paper from the provided text, which has been corrupted by initial PDF parsing.

**Characteristics of the Input Text:**
The input text likely contains the following types of undesirable content that **must be removed**:
1.  **PDF Formatting Residue:** Excessive line breaks, extra white spaces, pagination markers, incorrectly hyphenated words (e.g., "com-\nputer"), and rough text representations of tables or figures.
2.  **Non-Body Metadata:** Headers, footers, page numbers, running heads, copyright notices, conference statements (e.g., "Published at COLM 2025"), and arXiv identifiers.
3.  **Appended Non-Academic Commentary:** Any passages, philosophical discussions, or fictional commentary that are unrelated to the paper's core academic content (e.g., the large sections you mentioned about "assumed consent" or "Important Considerations").

**Task Instructions (Must be strictly followed):**
1.  **Only extract the academic core content:** This includes the Title, Authors, Abstract, Introduction, all main chapters/sections, Conclusions, Acknowledgements, and the complete References/Bibliography list. Ensure this academic content is preserved **in its entirety, word-for-word**.
2.  **Absolutely eliminate** all undesirable content listed under "Characteristics of the Input Text."
3.  **Formatting Requirements:**
    * Remove all pagination markers or page breaks (e.g., `--- Page 1 ---`).
    * Repair incorrectly hyphenated words (e.g., words broken by a hyphen `-` at the end of a line).
    * Use **two newline characters `\n\n`** to separate distinct paragraphs, ensuring the flow is clear and readable.
    * Remove any rough text-based representations of tables, figures, or complex mathematical formulas (e.g., table lines `|` or repetitive `---`).
    * **Maintain the original formatting of the section headings** (e.g., `1. Introduction`, `2. Methodology`).

**Output Requirement:**
Provide only the cleaned and extracted pure academic content from the paper. **Do not include any explanatory text, commentary, additional discussion outside of the cleaned paper text and any prefix words.**

**CRITICAL RULES:**
1. **Output ONLY the cleaned paper content** - nothing else
2. **DO NOT** add any introductory phrases like "Here is..." or "The cleaned content is:"
3. **DO NOT** add any explanatory text, comments, or summaries
4. **BEGIN DIRECTLY** with the paper's content

---
**Text to Process:**
PDF content: "{pdf_content}"
'''

def extract_text_from_local_pdf(file_path: str, clean_text: bool = True) -> Dict:
    try:
        print(f"Processing PDF: {file_path}")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text_content += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
            

            if clean_text and text_content:
                text_content = clean_extracted_text(text_content)
            
            result = {
                "success": True,
                "content": text_content.strip(),
                "content_length": len(text_content),
                "page_count": len(pdf_reader.pages)
            }
            
            print(f"Finished. Extract total {len(text_content)} char, {len(pdf_reader.pages)} pages")
            return result
            
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        return {"success": False, "error": f"Extract PDF fail: {str(e)}"}

def clean_extracted_text(text: str) -> str:

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = '\n'.join(line.strip() for line in text.split('\n'))

    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text.strip()



def get_paper_summary(paper_content):
    paper_summary_prompt = get_paper_summary_prompt(paper_content)
    messages = [{'role':'user','content':paper_summary_prompt}]

    for attempt in range(5):
        print(f"\n--- Paper Summary Generation: Attempt {attempt+1}/5 ---")
        paper_summary = remove_json_markdown(summerizer_generate(messages))
        try:
            paper_summary_obj = json.loads(paper_summary)
    
            required_fields = ['paper_title', 'paper_abstract', 'paper_methodology_summary']
            if all(field in paper_summary_obj for field in required_fields):
                print('Successfully obtained paper summary structure.')
                return paper_summary_obj['paper_title'], paper_summary_obj['paper_abstract'], paper_summary_obj['paper_methodology_summary']
            else:
                print(f"Summary JSON missing required fields: {', '.join([f for f in required_fields if f not in paper_summary_obj])}. Retrying...")
                messages.append({'role':'user','content':paper_output_format_instruction})
        except Exception as e:
            print('Error parsing summary JSON:')
            print(e)
            print('paper_summary is', paper_summary)
            messages.append({'role':'user','content':paper_output_format_instruction})

    return None, None, None

if __name__ == "__main__":
    import argparse
    out_json_path = config_loader.context_json_path

    parser = argparse.ArgumentParser(description='Process a PDF file and extract summary information.')
    parser.add_argument('--pdf_file_path', type=str, help='Path to the PDF file to process')
    args = parser.parse_args()
    
    pdf_file_path = args.pdf_file_path
    file_name = Path(pdf_file_path).stem

    os.makedirs('context', exist_ok=True)
    try:
        with open(out_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        data = {}

    if file_name not in data:

        result = extract_text_from_local_pdf(pdf_file_path)
    
        if result["success"]:
            print("\n" + "="*50)
            print("Paper Metadata:")
            print(f"Page Count: {result['page_count']}")
            print(f"Character Count: {result['content_length']}")
        

            paper = summerizer_generate([{'role':'user','content':get_clean_pdf_prompt(pdf_content = result['content'])}])
            paper_title, paper_abstract, paper_methodology  = get_paper_summary(paper)
            if paper_title is not None:
                data[file_name] = {
                    'paper_title':paper_title,
                    'paper_abstract':paper_abstract,
                    'paper_methodology':paper_methodology,
                    'full_content': paper
                }
                with open(out_json_path, 'w') as f:
                    json.dump(data, f, indent=1, ensure_ascii=False) 
            else:
                print(f"Processing Failed: can't get paper summary")
        else:
             print(f"Processing Failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"Paper already exists in database")