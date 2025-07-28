from datetime import datetime
import json
from pathlib import Path

def save_output(metadata, sections, refined, output_path):
    output = {
        "metadata": metadata,
        "extracted_sections": sections,
        "sub_section_analysis": refined
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
