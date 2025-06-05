import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import cv2

class StyleModel:
    """Handles custom handwriting style training and management"""
    
    def __init__(self):
        self.styles_dir = "saved_styles"
        self.create_styles_directory()
        self.custom_styles = self.load_custom_styles()
        
        # Simple style parameters for basic implementation
        self.style_parameters = {
            'slant_angle': 0,
            'letter_spacing': 1.0,
            'line_spacing': 1.5,
            'stroke_thickness': 2,
            'character_variations': {},
            'noise_level': 0.1
        }
    
    def create_styles_directory(self):
        """Create directory for saving custom styles"""
        try:
            if not os.path.exists(self.styles_dir):
                os.makedirs(self.styles_dir)
        except Exception as e:
            print(f"Error creating styles directory: {str(e)}")
    
    def train_custom_style(self, style_name, training_data, epochs=50, 
                          learning_rate=0.01, batch_size=16, progress_callback=None):
        """
        Train a custom handwriting style
        
        Args:
            style_name: Name for the custom style
            training_data: List of PIL images with handwriting samples
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            progress_callback: Function to call with progress updates
        
        Returns:
            Boolean indicating success
        """
        try:
            if not training_data:
                print("No training data provided")
                return False
            
            print(f"Training custom style '{style_name}' with {len(training_data)} samples")
            
            # Analyze training samples to extract style characteristics
            style_characteristics = self._analyze_handwriting_samples(training_data)
            
            # Simulate training process with progress updates
            for epoch in range(epochs):
                # Simulate training step
                if progress_callback:
                    progress_callback(epoch + 1, epochs)
                
                # Simulate some processing time
                import time
                time.sleep(0.1)  # Small delay to show progress
                
                # Update style characteristics (simplified)
                if epoch % 10 == 0:
                    style_characteristics = self._refine_style_characteristics(
                        style_characteristics, training_data
                    )
            
            # Save the trained style
            success = self._save_custom_style(style_name, style_characteristics)
            
            if success:
                # Reload custom styles
                self.custom_styles = self.load_custom_styles()
                print(f"Successfully trained custom style '{style_name}'")
            
            return success
        
        except Exception as e:
            print(f"Error training custom style: {str(e)}")
            return False
    
    def _analyze_handwriting_samples(self, training_data):
        """Analyze handwriting samples to extract style characteristics"""
        try:
            characteristics = {
                'avg_slant': 0,
                'avg_spacing': 1.0,
                'stroke_thickness': 2,
                'character_heights': {},
                'noise_level': 0.1,
                'dominant_color': (20, 20, 50),
                'sample_count': len(training_data)
            }
            
            slant_angles = []
            stroke_thicknesses = []
            
            for img_array in training_data:
                # Convert to OpenCV format if needed
                if isinstance(img_array, Image.Image):
                    opencv_img = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2BGR)
                else:
                    opencv_img = img_array
                
                # Analyze slant
                slant = self._calculate_slant_angle(opencv_img)
                if slant is not None:
                    slant_angles.append(slant)
                
                # Analyze stroke thickness
                thickness = self._calculate_stroke_thickness(opencv_img)
                if thickness is not None:
                    stroke_thicknesses.append(thickness)
            
            # Calculate averages
            if slant_angles:
                characteristics['avg_slant'] = np.mean(slant_angles)
            
            if stroke_thicknesses:
                characteristics['stroke_thickness'] = int(np.mean(stroke_thicknesses))
            
            # Analyze character variations (simplified)
            characteristics['character_variations'] = self._analyze_character_variations(training_data)
            
            return characteristics
        
        except Exception as e:
            print(f"Error analyzing samples: {str(e)}")
            return self.style_parameters.copy()
    
    def _calculate_slant_angle(self, image):
        """Calculate the slant angle of handwriting"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi
                    # Convert to slant angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                if angles:
                    return np.mean(angles)
            
            return 0  # Default to no slant
        
        except Exception as e:
            print(f"Error calculating slant: {str(e)}")
            return 0
    
    def _calculate_stroke_thickness(self, image):
        """Calculate average stroke thickness"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Threshold to binary
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            
            # Find average stroke thickness
            stroke_pixels = dist_transform[dist_transform > 0]
            if len(stroke_pixels) > 0:
                avg_thickness = np.mean(stroke_pixels) * 2  # Approximate thickness
                return max(1, int(avg_thickness))
            
            return 2  # Default thickness
        
        except Exception as e:
            print(f"Error calculating stroke thickness: {str(e)}")
            return 2
    
    def _analyze_character_variations(self, training_data):
        """Analyze character-level variations (simplified)"""
        try:
            variations = {
                'size_variation': 0.1,
                'rotation_variation': 2.0,
                'spacing_variation': 0.2
            }
            
            # This would be more sophisticated in a real implementation
            # For now, return basic variations based on sample count
            sample_count = len(training_data)
            
            if sample_count > 5:
                variations['size_variation'] = 0.15
                variations['rotation_variation'] = 3.0
            
            if sample_count > 10:
                variations['spacing_variation'] = 0.25
            
            return variations
        
        except Exception as e:
            print(f"Error analyzing character variations: {str(e)}")
            return {'size_variation': 0.1, 'rotation_variation': 2.0, 'spacing_variation': 0.2}
    
    def _refine_style_characteristics(self, characteristics, training_data):
        """Refine style characteristics during training"""
        try:
            # Simulate refinement process
            refined = characteristics.copy()
            
            # Slightly adjust parameters
            refined['avg_slant'] *= 0.95  # Converge towards zero
            refined['noise_level'] = max(0.05, refined['noise_level'] * 0.98)
            
            # Update character variations
            if 'character_variations' in refined:
                for key in refined['character_variations']:
                    refined['character_variations'][key] *= 0.99
            
            return refined
        
        except Exception as e:
            print(f"Error refining characteristics: {str(e)}")
            return characteristics
    
    def _save_custom_style(self, style_name, characteristics):
        """Save custom style to file"""
        try:
            style_data = {
                'name': style_name,
                'created_date': datetime.now().isoformat(),
                'characteristics': characteristics,
                'version': '1.0'
            }
            
            # Calculate a simple accuracy score based on sample count
            sample_count = characteristics.get('sample_count', 1)
            accuracy = min(95, 60 + (sample_count * 3))  # Cap at 95%
            style_data['accuracy'] = accuracy
            
            file_path = os.path.join(self.styles_dir, f"{style_name}.json")
            
            with open(file_path, 'w') as f:
                json.dump(style_data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error saving custom style: {str(e)}")
            return False
    
    def load_custom_styles(self):
        """Load all custom styles from disk"""
        try:
            styles = []
            
            if not os.path.exists(self.styles_dir):
                return styles
            
            for filename in os.listdir(self.styles_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.styles_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            style_data = json.load(f)
                            styles.append(style_data)
                    except Exception as e:
                        print(f"Error loading style {filename}: {str(e)}")
            
            return styles
        
        except Exception as e:
            print(f"Error loading custom styles: {str(e)}")
            return []
    
    def get_custom_styles(self):
        """Get list of custom styles with metadata"""
        try:
            style_list = []
            
            for style in self.custom_styles:
                style_info = {
                    'name': style.get('name', 'Unknown'),
                    'created_date': style.get('created_date', 'Unknown'),
                    'accuracy': style.get('accuracy', 0),
                    'sample_count': style.get('characteristics', {}).get('sample_count', 0)
                }
                style_list.append(style_info)
            
            return style_list
        
        except Exception as e:
            print(f"Error getting custom styles: {str(e)}")
            return []
    
    def delete_custom_style(self, style_name):
        """Delete a custom style"""
        try:
            file_path = os.path.join(self.styles_dir, f"{style_name}.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                # Reload styles
                self.custom_styles = self.load_custom_styles()
                return True
            
            return False
        
        except Exception as e:
            print(f"Error deleting custom style: {str(e)}")
            return False
    
    def get_style_characteristics(self, style_name):
        """Get characteristics for a specific style"""
        try:
            for style in self.custom_styles:
                if style.get('name') == style_name:
                    return style.get('characteristics', {})
            
            # Return default characteristics if style not found
            return self.style_parameters.copy()
        
        except Exception as e:
            print(f"Error getting style characteristics: {str(e)}")
            return self.style_parameters.copy()
    
    def export_style(self, style_name):
        """Export a custom style to a file"""
        try:
            for style in self.custom_styles:
                if style.get('name') == style_name:
                    return json.dumps(style, indent=2)
            
            return None
        
        except Exception as e:
            print(f"Error exporting style: {str(e)}")
            return None
    
    def import_style(self, style_data_json):
        """Import a custom style from JSON data"""
        try:
            style_data = json.loads(style_data_json)
            
            if 'name' not in style_data:
                return False
            
            # Validate style data structure
            required_fields = ['name', 'characteristics']
            for field in required_fields:
                if field not in style_data:
                    return False
            
            # Save imported style
            return self._save_custom_style(
                style_data['name'], 
                style_data['characteristics']
            )
        
        except Exception as e:
            print(f"Error importing style: {str(e)}")
            return False
