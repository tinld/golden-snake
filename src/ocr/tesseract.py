import os
from pathlib import Path
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import re
import logging
from typing import List
from src.media.pdf_processor import PDFDocumentProcessor, PDFChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
      
      def transform_pdf_images_to_chunk_text(
            self, 
            pdf_path: str,
            chunk_size: int = 500,
            chunk_overlap: int = 50
      ) -> List[str]:
            """
            Extract text from PDF images using OCR and chunk into meaningful segments.
            
            Args:
                pdf_path: Path to PDF file
                chunk_size: Target chunk size in characters (default: 500)
                chunk_overlap: Character overlap between chunks (default: 50)
                
            Returns:
                List of text chunks suitable for embedding
            """
            images = convert_from_path(pdf_path)
            full_text = ""
            
            # Extract text from all images
            for img in images:
                  text = pytesseract.image_to_string(
                        img,
                        lang="vie",
                        config="--oem 3 --psm 6"
                  )
                  if text.strip():
                        full_text += text + "\n"

            if not full_text.strip():
                  raise ValueError("No text extracted from PDF images.")
            
            # Use PDFChunker for intelligent chunking
            chunker = PDFChunker(
                  chunk_size=chunk_size,
                  chunk_overlap=chunk_overlap
            )
            
            # Clean and chunk the full text
            chunk_texts = chunker.chunk_text(full_text)
            
            return chunk_texts
      
      def run_test_pdf(
            self, 
            pdf_path: str, 
            lang: str = 'vie', 
            config: str = '--oem 3 --psm 6',
            chunk_size: int = 500,
            chunk_overlap: int = 50
      ) -> dict:
            """
            Extract text from PDF images using OCR and return intelligent chunks.
            
            Args:
                pdf_path: Path to the PDF file
                lang: Language code for OCR (default: 'vie' for Vietnamese)
                config: Tesseract config string (default: '--oem 3 --psm 6')
                chunk_size: Target chunk size in characters
                chunk_overlap: Character overlap between chunks
                
            Returns:
                Dictionary with chunk information and statistics
            """
            
            try:
                  # Verify PDF file exists
                  if not os.path.exists(pdf_path):
                        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                  
                  # Extract and chunk text from PDF images
                  chunks = self.transform_pdf_images_to_chunk_text(
                        pdf_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                  )
                  
                  # Filter and validate chunks
                  valid_chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) >= 20]
                  
                  # Generate statistics
                  stats = {
                        "total_chunks": len(valid_chunks),
                        "avg_length": sum(len(c) for c in valid_chunks) // len(valid_chunks) if valid_chunks else 0,
                        "min_length": min((len(c) for c in valid_chunks), default=0),
                        "max_length": max((len(c) for c in valid_chunks), default=0),
                        "total_chars": sum(len(c) for c in valid_chunks)
                  }
                  
                  logger.info(f"OCR chunking complete: {stats['total_chunks']} chunks, avg {stats['avg_length']} chars")
                  
                  return {
                        "status": "success",
                        "chunks": valid_chunks,
                        "statistics": stats,
                        "pdf_path": pdf_path
                  }
                  
            except Exception as e:
                  error_msg = f"Error processing PDF with OCR: {str(e)}"
                  logger.error(error_msg)
                  return {
                        "status": "error",
                        "message": error_msg,
                        "pdf_path": pdf_path
                  }
                  