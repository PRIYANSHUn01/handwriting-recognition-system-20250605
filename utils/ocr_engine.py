import cv2
import numpy as np
import pytesseract
import re
from PIL import Image

class OCREngine:
    """Handles OCR functionality using Tesseract"""
    
    def __init__(self):
        # Configure Tesseract path if needed (adjust based on system)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        
        self.default_config = r'--oem 3 --psm 6'
        self.language = 'eng'
    
    def extract_text(self, image):
        """
        Extract text from preprocessed image
        
        Args:
            image: OpenCV image (numpy array)
        
        Returns:
            Dictionary containing extracted text, confidence, and word details
        """
        try:
            # Convert OpenCV image to PIL format
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Extract text with confidence
            text_data = pytesseract.image_to_data(
                pil_image, 
                config=self.default_config,
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract plain text
            extracted_text = pytesseract.image_to_string(
                pil_image,
                config=self.default_config,
                lang=self.language
            ).strip()
            
            # Calculate overall confidence
            confidences = [int(conf) for conf in text_data['conf'] if int(conf) > 0]
            overall_confidence = np.mean(confidences) if confidences else 0
            
            # Extract word-level details
            word_details = self._extract_word_details(text_data)
            
            # Clean and post-process text
            cleaned_text = self._post_process_text(extracted_text)
            
            return {
                'text': cleaned_text,
                'confidence': overall_confidence,
                'word_details': word_details,
                'raw_text': extracted_text
            }
        
        except Exception as e:
            print(f"Error in OCR extraction: {str(e)}")
            return {
                'text': '',
                'confidence': 0,
                'word_details': [],
                'error': str(e)
            }
    
    def _extract_word_details(self, text_data):
        """Extract word-level details from Tesseract output"""
        word_details = []
        
        try:
            n_boxes = len(text_data['text'])
            
            for i in range(n_boxes):
                if int(text_data['conf'][i]) > 0:  # Only include confident detections
                    word_text = text_data['text'][i].strip()
                    if word_text:  # Only include non-empty text
                        word_details.append({
                            'text': word_text,
                            'confidence': float(text_data['conf'][i]),
                            'bbox': [
                                int(text_data['left'][i]),
                                int(text_data['top'][i]),
                                int(text_data['width'][i]),
                                int(text_data['height'][i])
                            ]
                        })
        
        except Exception as e:
            print(f"Error extracting word details: {str(e)}")
        
        return word_details
    
    def _post_process_text(self, text):
        """Clean and post-process extracted text"""
        try:
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', text)
            
            # Remove special characters that are likely OCR errors
            cleaned = re.sub(r'[^\w\s\.,!?;:\-\'"()]', '', cleaned)
            
            # Fix common OCR mistakes
            replacements = {
                r'\b0\b': 'O',  # Zero to O
                r'\b1\b(?=\w)': 'I',  # 1 to I when followed by letters
                r'\b5\b(?=\w)': 'S',  # 5 to S when followed by letters
                r'\|': 'l',  # Pipe to lowercase l
                r'rn': 'm',  # rn to m
                r'vv': 'w',  # vv to w
            }
            
            for pattern, replacement in replacements.items():
                cleaned = re.sub(pattern, replacement, cleaned)
            
            # Capitalize first letter of sentences
            sentences = re.split(r'[.!?]+', cleaned)
            capitalized_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    capitalized_sentences.append(sentence)
            
            if capitalized_sentences:
                cleaned = '. '.join(capitalized_sentences)
                if not cleaned.endswith('.'):
                    cleaned += '.'
            
            return cleaned.strip()
        
        except Exception as e:
            print(f"Error in text post-processing: {str(e)}")
            return text
    
    def extract_text_with_layout(self, image):
        """Extract text while preserving layout information"""
        try:
            # Convert OpenCV image to PIL format
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Use PSM 6 for uniform block of text
            layout_data = pytesseract.image_to_data(
                pil_image,
                config=r'--oem 3 --psm 6',
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Group text by lines
            lines = {}
            for i in range(len(layout_data['text'])):
                if int(layout_data['conf'][i]) > 30:  # Filter low confidence
                    text = layout_data['text'][i].strip()
                    if text:
                        line_num = layout_data['line_num'][i]
                        if line_num not in lines:
                            lines[line_num] = []
                        lines[line_num].append({
                            'text': text,
                            'x': layout_data['left'][i],
                            'y': layout_data['top'][i],
                            'confidence': layout_data['conf'][i]
                        })
            
            # Reconstruct text with line breaks
            formatted_text = []
            for line_num in sorted(lines.keys()):
                line_words = sorted(lines[line_num], key=lambda x: x['x'])
                line_text = ' '.join([word['text'] for word in line_words])
                formatted_text.append(line_text)
            
            return '\n'.join(formatted_text)
        
        except Exception as e:
            print(f"Error in layout extraction: {str(e)}")
            return self.extract_text(image)['text']
    
    def set_language(self, language_code):
        """Set OCR language"""
        try:
            self.language = language_code
            return True
        except Exception as e:
            print(f"Error setting language: {str(e)}")
            return False
    
    def get_available_languages(self):
        """Get list of available Tesseract languages"""
        try:
            languages = pytesseract.get_languages()
            return languages
        except Exception as e:
            print(f"Error getting languages: {str(e)}")
            return ['eng']  # Default fallback
    
    def extract_text_regions(self, image, regions):
        """Extract text from specific regions of the image"""
        try:
            results = []
            
            for i, (x, y, w, h) in enumerate(regions):
                # Extract region
                roi = image[y:y+h, x:x+w]
                
                # Extract text from region
                region_result = self.extract_text(roi)
                region_result['region_id'] = i
                region_result['bbox'] = [x, y, w, h]
                
                results.append(region_result)
            
            return results
        
        except Exception as e:
            print(f"Error extracting text regions: {str(e)}")
            return []
    
    def validate_text_quality(self, text, min_confidence=50):
        """Validate extracted text quality"""
        try:
            # Basic quality checks
            quality_score = 100
            
            # Check for excessive special characters
            special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
            if special_char_ratio > 0.3:
                quality_score -= 30
            
            # Check for excessive single characters
            single_chars = len(re.findall(r'\b\w\b', text))
            if single_chars > len(text.split()) * 0.5:
                quality_score -= 20
            
            # Check for reasonable word length distribution
            words = text.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if avg_word_length < 2 or avg_word_length > 15:
                    quality_score -= 15
            
            return {
                'quality_score': max(0, quality_score),
                'is_valid': quality_score >= min_confidence,
                'special_char_ratio': special_char_ratio,
                'word_count': len(words) if words else 0,
                'avg_word_length': avg_word_length if words else 0
            }
        
        except Exception as e:
            print(f"Error validating text quality: {str(e)}")
            return {'quality_score': 0, 'is_valid': False}
