import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
import os
import docx

class LoadData:
    def __read_pdf(file_path):
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text

    def __read_word(file_path):
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def __read_txt(file_path,encoding="utf-8"):
        with open(file_path, "r",encoding=encoding) as file:
            text = file.read()
        return text
    
    def __read_csv(file_path,encoding="utf-8"):
        return pd.read_csv(file_path,encoding=encoding)
    
    def clean_text(text):
        return re.sub(r'\n+', '\n', text).strip()
    
    def read_documents_from_directory(directory,encoding="utf-8"):
        combined_text = ""
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".pdf"):
                combined_text += LoadData.__read_pdf(file_path)
            elif filename.endswith(".docx"):
                combined_text += LoadData.__read_word(file_path)
            elif filename.endswith(".txt"):
                combined_text += LoadData.__read_txt(file_path,encoding)
        return combined_text
    
    def read_multiple_files(file_paths):
        combined_text = ""
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                combined_text += LoadData.__read_pdf(file_path)
            elif file_path.endswith(".docx"):
                combined_text += LoadData.__read_word(file_path)
            elif file_path.endswith(".txt"):
                combined_text += LoadData.__read_txt(file_path)
        return combined_text
    
    def read_csv(file_path,encoding="utf-8"):
        return LoadData.__read_csv(file_path,encoding)