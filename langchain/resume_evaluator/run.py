
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
                suffix = os.path.splitext(filename)[1].lower()
                with open(file_path, 'r', encoding='utf-8') as file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.file.read())
                        tmp.flush()
                        if suffix == '.pdf':
                            print(f"Reading pdf file {filename}")
                            # with pdfplumber.open(tmp.name) as pdf:
                            #     return "\n".join(page.extract_text() or "" for page in pdf.pages)
                        elif suffix in ['.docx', '.doc']:
                            print(f"Reading Eord document file {filename}")
                            # doc = docx.Document(tmp.name)
                            # return "\n".join([para.text for para in doc.paragraphs])
                        else:
                            return tmp.read().decode('utf-8', errors='ignore')

            except Exception as e:
                print(f"Could not read file {filename}: {e}")



if __name__ == "__main__":
    # Usage
    read_all_files('./resume_files')
