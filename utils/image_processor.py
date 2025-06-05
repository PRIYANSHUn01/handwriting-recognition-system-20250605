import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    """Handles image preprocessing for better OCR accuracy"""
    
    def __init__(self):
        self.processing_methods = {
            'noise_reduction': self._apply_noise_reduction,
            'binarization': self._apply_binarization,
            'contrast_enhancement': self._enhance_contrast,
            'skew_correction': self._correct_skew,
            'edge_enhancement': self._enhance_edges
        }
    
    def preprocess_image(self, image, **kwargs):
        """
        Apply various preprocessing techniques to improve OCR accuracy
        
        Args:
            image: OpenCV image (numpy array)
            **kwargs: Processing options (noise_reduction, binarization, etc.)
        
        Returns:
            Preprocessed OpenCV image
        """
        processed_image = image.copy()
        
        try:
            # Apply requested preprocessing steps
            if kwargs.get('noise_reduction', False):
                processed_image = self._apply_noise_reduction(processed_image)
            
            if kwargs.get('contrast_enhancement', False):
                processed_image = self._enhance_contrast(processed_image)
            
            if kwargs.get('binarization', False):
                processed_image = self._apply_binarization(processed_image)
            
            if kwargs.get('skew_correction', False):
                processed_image = self._correct_skew(processed_image)
            
            if kwargs.get('edge_enhancement', False):
                processed_image = self._enhance_edges(processed_image)
            
            return processed_image
        
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            return image
    
    def _apply_noise_reduction(self, image):
        """Apply noise reduction using bilateral filter"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Convert back to BGR if original was color
            if len(image.shape) == 3:
                return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            else:
                return denoised
        
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return image
    
    def _apply_binarization(self, image):
        """Apply adaptive thresholding for binarization"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to BGR
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        except Exception as e:
            print(f"Error in binarization: {str(e)}")
            return image
    
    def _enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        try:
            # Convert to LAB color space
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
            else:
                l_channel = image.copy()
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels and convert back
            if len(image.shape) == 3:
                enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced = l_channel
            
            return enhanced
        
        except Exception as e:
            print(f"Error in contrast enhancement: {str(e)}")
            return image
    
    def _correct_skew(self, image):
        """Correct skew in the image"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Rotate image if significant skew detected
                    if abs(avg_angle) > 0.5:
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                        corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                                 flags=cv2.INTER_CUBIC, 
                                                 borderMode=cv2.BORDER_REPLICATE)
                        return corrected
            
            return image
        
        except Exception as e:
            print(f"Error in skew correction: {str(e)}")
            return image
    
    def _enhance_edges(self, image):
        """Enhance edges in the image"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply unsharp masking for edge enhancement
            blurred = cv2.GaussianBlur(gray, (0, 0), 2)
            enhanced = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # Convert back to BGR if needed
            if len(image.shape) == 3:
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                return enhanced
        
        except Exception as e:
            print(f"Error in edge enhancement: {str(e)}")
            return image
    
    def get_image_stats(self, image):
        """Get basic statistics about the image"""
        try:
            stats = {}
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            stats['dimensions'] = image.shape[:2]
            stats['mean_intensity'] = np.mean(gray)
            stats['std_intensity'] = np.std(gray)
            stats['min_intensity'] = np.min(gray)
            stats['max_intensity'] = np.max(gray)
            
            # Calculate contrast (RMS contrast)
            stats['contrast'] = np.sqrt(np.mean((gray - np.mean(gray)) ** 2))
            
            # Estimate noise level
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            stats['sharpness'] = laplacian_var
            
            return stats
        
        except Exception as e:
            print(f"Error calculating image stats: {str(e)}")
            return {}
    
    def resize_for_ocr(self, image, target_height=800):
        """Resize image to optimal size for OCR"""
        try:
            height, width = image.shape[:2]
            
            if height < target_height:
                # Scale up small images
                scale_factor = target_height / height
                new_width = int(width * scale_factor)
                resized = cv2.resize(image, (new_width, target_height), 
                                   interpolation=cv2.INTER_CUBIC)
            elif height > target_height * 2:
                # Scale down very large images
                scale_factor = target_height / height
                new_width = int(width * scale_factor)
                resized = cv2.resize(image, (new_width, target_height), 
                                   interpolation=cv2.INTER_AREA)
            else:
                resized = image.copy()
            
            return resized
        
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return image
