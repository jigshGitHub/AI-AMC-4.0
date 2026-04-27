
import os
import pdfplumber
import docx
import tempfile

def read_all_files(folder_path):
    # Ensure the path exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return

    # Loop through every file in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (and not a subfolder)
        if os.path.isfile(file_path):
            try:
                # with open(file_path, 'r', encoding='utf-8') as file:
                #     content = file.read()
                #     print(f"--- Content of {filename} ---")
                #     print(content)
                #     print("-" * 20)
                texts = ""
                if filename.endswith('.pdf'):
                    print(f"Reading pdf file {filename}")
                    with pdfplumber.open(file_path) as pdf:
                        texts = "\n".join(page.extract_text() or "" for page in pdf.pages)
                elif filename.endswith('.docx'):
                    print(f"Reading Eord document file {filename}")
                    doc = docx.Document(file_path)
                    texts = "\n".join([para.text for para in doc.paragraphs])
                print(texts[:200])
            except Exception as e:
                print(f"Could not read file {filename}: {e}")



if __name__ == "__main__":
    # Usage
    read_all_files('./resume_files')
