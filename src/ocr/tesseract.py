import os
from pathlib import Path
import pytesseract
from PIL import Image


class TesseractOCR:
      def __init__(self, tesseract_cmd: str = None, tessdata_dir: str = None):
            self.name = "Tesseract OCR"
            
            # Set tesseract command path if provided
            if tesseract_cmd:
                  pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            elif os.name == 'nt':  # Windows
                  # Try common Windows installation paths
                  common_paths = [
                        r'C:\Users\tinld\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
                  ]
                  for path in common_paths:
                        if os.path.exists(path):
                              pytesseract.pytesseract.tesseract_cmd = path
                              break
            
            # Set TESSDATA_PREFIX environment variable
            if tessdata_dir:
                  os.environ['TESSDATA_PREFIX'] = tessdata_dir
            elif os.name == 'nt':  # Windows
                  # Try common tessdata paths
                  tessdata_paths = [
                        r'C:\Users\tinld\AppData\Local\Programs\Tesseract-OCR\tessdata'
                  ]
                  for path in tessdata_paths:
                        if os.path.exists(path):
                              os.environ['TESSDATA_PREFIX'] = path
                              break
  
      def run_test_image(self, image_path: str, lang: str = 'eng', config: str = '') -> str:
            """
            Extract text from an image using Tesseract OCR.
            
            Args:
                image_path: Path to the image file
                lang: Language code for OCR (default: 'eng' for English)
                      Multiple languages can be specified like 'eng+vie'
                config: Additional Tesseract configuration string
            
            Returns:
                Extracted text from the image
                
            Raises:
                FileNotFoundError: If image file doesn't exist
                Exception: If OCR processing fails
            """
            try:
                  # Verify image file exists
                  if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image file not found: {image_path}")
                  
                  # Open and process the image
                  image = Image.open(image_path)
                  
                  # Perform OCR
                  text = pytesseract.image_to_string(image, lang=lang, config=config)
                  
                  return text.strip()
            except Exception as e:
                  raise Exception(f"Error processing image with OCR: {str(e)}")
      
      def get_available_languages(self) -> list:
            """
            Get list of available languages installed in Tesseract.
            
            Returns:
                List of language codes
            """
            try:
                  langs = pytesseract.get_languages()
                  return langs
            except Exception as e:
                  print(f"Error getting available languages: {e}")
                  return []
      
      def run_test_pdf(self, pdf_path: str, lang: str = 'eng', config: str = '') -> str:
            """
            Extract text from an image and save as searchable PDF using Tesseract OCR.
            
            Args:
                image_path: Path to the image file
                lang: Language code for OCR (default: 'eng' for English)
                      Multiple languages can be specified like 'eng+vie'
            """
            
            try:
                  # Verify image file exists
                  if not os.path.exists(pdf_path):
                        raise FileNotFoundError(f"Image file not found: {pdf_path}")
                  
                  # Open and process the pdf image
                  image = Image.open(pdf_path)
                  
                  
            except Exception as e:
                  raise Exception(f"Error processing image to PDF with OCR: {str(e)}")