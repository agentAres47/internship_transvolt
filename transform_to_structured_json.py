import os
import json
from groq import Groq

os.environ["GROQ_API_KEY"] = "" #your api key

# Load Groq API key
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def transform_invoice_data(raw_json, schema):
    prompt = f"""
    Your job is to reformat invoice data (from OCR) into a fixed schema.

    - Translate non-English to English.
    - Use null for missing values.
    - Only include what's in the schema.
    - Format dates as YYYY-MM-DD where possible.

    SCHEMA:
    {json.dumps(schema, indent=2)}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "text", "text": f"RAW JSON:\n{json.dumps(raw_json)}"}]}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

def main():
    extracted_dir = "./outputs"
    schema_path = "./invoice_target_schema.json"
    output_dir = "./structured_outputs"
    os.makedirs(output_dir, exist_ok=True)

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    for file in os.listdir(extracted_dir):
        if file.endswith(".json"):
            path = os.path.join(extracted_dir, file)
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            # Handle if JSON is a list of pages
            if isinstance(raw, list):
                raw = {k: v for d in raw for k, v in d.items()}

            try:
                print(f"🔁 Structuring: {file}")
                structured = transform_invoice_data(raw, schema)
                out_file = os.path.join(output_dir, f"structured_{file}")
                with open(out_file, 'w', encoding='utf-8') as f_out:
                    json.dump(structured, f_out, indent=2, ensure_ascii=False)
                print(f" Saved: {out_file}")
            except Exception as e:
                print(f" Failed on {file}: {e}")

if __name__ == "__main__":
    print("🚀 Transforming OCR JSONs to structured format...")
    main()
