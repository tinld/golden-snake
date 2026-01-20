import sys
import logging
from pathlib import Path
from src.ocr.tesseract import TesseractOCR

sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_tesseract():
      """Test Tesseract OCR functionality"""
      print("\n[1] Initializing Tesseract OCR...")
      ocr = TesseractOCR()
      texts = ocr.run(image_path='media/test.jpg', lang='vie')
      print("[2] Extracted Text:")
      print(texts)
      return

if __name__ == "__main__":
    success = test_tesseract()
    if success:
        print("\n✓ PDF saving test completed successfully")
    else:
        print("\n✗ PDF saving test failed")