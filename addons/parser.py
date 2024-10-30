import io
#import pdfplumber
#import mimetypes
import os
import pypandoc
from pdf2image import convert_from_path
#from PIL import Image
from tika import parser, initVM
#from docx import Document
#from pptx import Presentation
#from openpyxl import load_workbook
from fpdf import FPDF

import asyncio
import aiofiles
import subprocess

#import ollama

#initVM()
#async def extract(filemap: dict):
#    file = filemap['content']
#    parsed = parser.from_buffer(file)
#    
#    print(parsed)
#    
#    mime_type, _ = mimetypes.guess_type(filemap['name'])
#    if not mime_type:
#        raise ValueError("Unable to determine file extension")
#    if not parsed:
#        raise ValueError("File could not be loaded")
#    
#    metadata = parsed['metadata']
#    text = ""
#    if 'content' in parsed:
#        text = parsed['content']
#    
#    if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
#        file = io.BytesIO(file)
#        doc = Document(file)
#        paragraphs = []
#        for para in doc.paragraphs:
#            paragraphs.append(para.text)
#        text = '\n'.join(paragraphs)
#
#    # If the file is a PDF, use pdfplumber to parse it
#    elif mime_type == 'application/pdf':
#        file = io.BytesIO(file)
#        with pdfplumber.open(file) as pdf:
#            pages = []
#            for page in pdf.pages:
#                pages.append(page.extract_text())
#            text = '\n\n'.join(pages)
#    
#    elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
#        file = io.BytesIO(file)
#        wb = load_workbook(filename=file, read_only=True, keep_vba=False)
#        ws = wb.active
#        text = ''
#        for row in ws.rows:
#            for cell in row:
#                if isinstance(cell.value, str):
#                    text += str(cell.value) + ' '
#        
#    elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
#        file = io.BytesIO(file)
#        prs = Presentation(file)
#        text = ""
#        for count, slide in enumerate(prs.slides):
#            text += f"Slide {count}:\n\n"
#            for shape in slide.shapes:
#                if hasattr(shape, "text"):
#                    text += shape.text + "\n"
#
#    return metadata, text
#

initVM()
async def extract(filemap: dict):
    file_name = filemap['name']
    file_content = filemap['content']
    
    office_ext = {'.docx', '.doc', '.odt', '.rtf', '.epub', '.xlsx', '.xls', '.csv', '.pptx', '.ppt'}
    pandoc_ext = frozenset(map(lambda x: f".{x}", pypandoc.get_pandoc_formats()))
    
    file_ext = os.path.splitext(file_name)[1].lower()
    file_title = os.path.splitext(file_name)[0]
    
    temp_inp = f"{file_name}_temp{file_ext}"
    temp_out = f"{file_title}.pdf"
    
    async with aiofiles.open(temp_inp, mode='wb') as tempfile:
        await tempfile.write(file_content)
    
    try:
        if file_ext in office_ext:
            process = await asyncio.create_subprocess_exec(
                'libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(temp_out), temp_inp,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stderr)
            
            base_name = os.path.splitext(os.path.basename(temp_inp))[0]
            generated_pdf = os.path.join(os.path.dirname(temp_out), f"{base_name}.pdf")

            if generated_pdf != temp_out:
                os.rename(generated_pdf, temp_out)
            
        elif file_ext in pandoc_ext:
            pypandoc.convert_file(temp_inp, 'pdf', outputfile=temp_out)
        elif file_ext != '.pdf':
            parsed = parser.from_buffer(file_content)
            text = ""
            if 'content' in parsed:
                text = parsed['content']
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            
            for line in text.split('\n'):
                pdf.cell(200, 10, txt=line, ln=True)
            
            pdf.output(temp_out)
        elif file_ext == '.pdf':
            async with aiofiles.open(temp_out, mode='wb') as tempfile:
                await tempfile.write(file_content)
        
        parsed = parser.from_file(temp_out)
        
        imgs = convert_from_path(temp_out, dpi=200)
        
        for img in imgs:
            btio = io.BytesIO()
            img.save(btio, format="PNG")
        
        img_bytes = btio.getvalue()
        
        return img_bytes, parsed['content']
    
    finally:
        if os.path.exists(temp_inp):
            os.remove(temp_inp)
        if os.path.exists(temp_out):
            os.remove(temp_out)

#async def file_to_byte_array(file_path):
#    if not os.path.isfile(file_path):
#        raise FileNotFoundError(f"File '{file_path}' does not exist.")
#    
#    with open(file_path, 'rb') as file:
#        byte_array = bytearray(file.read())
#        
#    return byte_array
#
#
#async def main():
#    name = 'fortesting/Get_Started_With_Smallpdf-output.pdf'
#    byte = bytes(await file_to_byte_array(name))
#    res = await extractimg({'name': name, 'content': byte})
#    file_ext = os.path.splitext(name)[1].lower()
#    async for part in await ollama.AsyncClient().generate(model='llava', prompt=f"Analyze the {file_ext} file and describe its contents. Assume this is the text of the file: {res[1]}", images=[res[0]], stream=True):
#        print(part['response'], end='', flush=True)
#    print()
#
#if __name__ == '__main__':
#    asyncio.run(main())