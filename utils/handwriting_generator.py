import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math
from data.fonts.handwriting_styles import HandwritingStyles

class HandwritingGenerator:
    """Generates synthetic handwritten text"""
    
    def __init__(self):
        self.handwriting_styles = HandwritingStyles()
        self.default_style = 'casual'
        
        # Character variation parameters
        self.char_variations = {
            'rotation_range': (-5, 5),  # degrees
            'scale_range': (0.9, 1.1),
            'offset_range': (-2, 2),    # pixels
            'thickness_range': (-1, 1)
        }
        
        # Noise and texture parameters
        self.noise_params = {
            'ink_blobs': 0.02,          # probability of ink blobs
            'paper_texture': 0.1,       # paper texture intensity
            'fade_probability': 0.05     # probability of faded characters
        }
    
    def generate_handwriting(self, text, style='casual', **kwargs):
        """
        Generate handwritten text image
        
        Args:
            text: Input text to convert
            style: Handwriting style name
            **kwargs: Generation parameters
        
        Returns:
            PIL Image of generated handwriting
        """
        try:
            # Get parameters
            font_size = kwargs.get('font_size', 20)
            line_spacing = kwargs.get('line_spacing', 1.5)
            letter_spacing = kwargs.get('letter_spacing', 1.0)
            add_noise = kwargs.get('add_noise', True)
            ink_thickness = kwargs.get('ink_thickness', 2)
            
            # Split text into lines
            lines = text.split('\n')
            if not lines:
                return None
            
            # Calculate image dimensions
            estimated_width = max(len(line) for line in lines) * font_size * 0.6
            estimated_height = len(lines) * font_size * line_spacing + 100
            
            # Create canvas
            img_width = max(800, int(estimated_width * 1.2))
            img_height = max(200, int(estimated_height))
            
            # Create image with paper-like background
            image = self._create_paper_background(img_width, img_height)
            draw = ImageDraw.Draw(image)
            
            # Get font configuration for style
            font_config = self.handwriting_styles.get_style_config(style)
            
            # Generate each line
            y_position = 50
            for line in lines:
                if line.strip():  # Skip empty lines
                    self._draw_handwritten_line(
                        draw, line, y_position, font_size,
                        letter_spacing, font_config, add_noise, ink_thickness
                    )
                y_position += int(font_size * line_spacing)
            
            # Add paper texture and aging if requested
            if add_noise:
                image = self._add_paper_effects(image)
            
            return image
        
        except Exception as e:
            print(f"Error generating handwriting: {str(e)}")
            return None
    
    def _create_paper_background(self, width, height):
        """Create paper-like background"""
        try:
            # Create off-white background
            background_color = (250, 248, 245)  # Slightly off-white
            image = Image.new('RGB', (width, height), background_color)
            
            # Add subtle paper texture
            pixels = np.array(image)
            noise = np.random.normal(0, 3, (height, width, 3))
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            
            return Image.fromarray(pixels)
        
        except Exception as e:
            print(f"Error creating background: {str(e)}")
            return Image.new('RGB', (width, height), (255, 255, 255))
    
    def _draw_handwritten_line(self, draw, text, y_pos, font_size, letter_spacing, 
                              font_config, add_noise, ink_thickness):
        """Draw a line of handwritten text"""
        try:
            x_position = 50  # Left margin
            
            for char in text:
                if char == ' ':
                    x_position += font_size * 0.3  # Space width
                    continue
                
                # Get character variations if noise is enabled
                if add_noise:
                    char_variations = self._get_character_variations()
                else:
                    char_variations = {'rotation': 0, 'scale': 1.0, 'offset': (0, 0)}
                
                # Draw character with variations
                char_x = x_position + char_variations['offset'][0]
                char_y = y_pos + char_variations['offset'][1]
                
                # Simulate handwriting by drawing character path
                self._draw_handwritten_character(
                    draw, char, char_x, char_y, font_size,
                    char_variations, font_config, ink_thickness
                )
                
                # Move to next character position
                char_width = self._get_character_width(char, font_size)
                x_position += char_width * letter_spacing
        
        except Exception as e:
            print(f"Error drawing line: {str(e)}")
    
    def _draw_handwritten_character(self, draw, char, x, y, font_size, 
                                   variations, font_config, ink_thickness):
        """Draw individual character with handwriting effects"""
        try:
            # Get character stroke pattern
            strokes = self._get_character_strokes(char, font_size)
            
            # Apply variations
            if variations['rotation'] != 0:
                strokes = self._rotate_strokes(strokes, variations['rotation'])
            
            if variations['scale'] != 1.0:
                strokes = self._scale_strokes(strokes, variations['scale'])
            
            # Draw strokes
            ink_color = font_config.get('color', (20, 20, 50))  # Dark blue-black
            
            for stroke in strokes:
                # Add slight randomness to stroke
                stroke_points = [(x + p[0], y + p[1]) for p in stroke]
                
                if len(stroke_points) > 1:
                    # Draw stroke as connected lines
                    for i in range(len(stroke_points) - 1):
                        start_point = stroke_points[i]
                        end_point = stroke_points[i + 1]
                        
                        # Vary line thickness slightly
                        thickness = ink_thickness + random.randint(-1, 1)
                        thickness = max(1, thickness)
                        
                        draw.line([start_point, end_point], fill=ink_color, width=thickness)
        
        except Exception as e:
            print(f"Error drawing character '{char}': {str(e)}")
            # Fallback: draw simple text
            try:
                draw.text((x, y), char, fill=(20, 20, 50))
            except:
                pass
    
    def _get_character_strokes(self, char, font_size):
        """Get stroke patterns for character (simplified version)"""
        try:
            # This is a simplified implementation
            # In a full implementation, you'd have detailed stroke data for each character
            
            # For now, create basic stroke patterns
            strokes = []
            base_size = font_size * 0.6
            
            if char.isalpha():
                if char.lower() in 'aeiou':
                    # Vowels - circular patterns
                    strokes.append(self._create_circular_stroke(base_size))
                else:
                    # Consonants - linear patterns
                    strokes.append(self._create_linear_stroke(base_size))
            elif char.isdigit():
                # Numbers
                strokes.append(self._create_number_stroke(char, base_size))
            else:
                # Punctuation and special characters
                strokes.append(self._create_punctuation_stroke(char, base_size))
            
            return strokes
        
        except Exception as e:
            print(f"Error getting strokes for '{char}': {str(e)}")
            return [[(0, 0), (font_size//2, font_size//2)]]  # Fallback simple stroke
    
    def _create_circular_stroke(self, size):
        """Create circular stroke pattern for vowels"""
        points = []
        center_x, center_y = size // 2, size // 2
        radius = size * 0.3
        
        for angle in range(0, 360, 15):
            x = center_x + radius * math.cos(math.radians(angle))
            y = center_y + radius * math.sin(math.radians(angle))
            # Add slight randomness
            x += random.uniform(-1, 1)
            y += random.uniform(-1, 1)
            points.append((int(x), int(y)))
        
        return points
    
    def _create_linear_stroke(self, size):
        """Create linear stroke pattern for consonants"""
        points = []
        
        # Create a slightly curved line
        start_x, start_y = 0, size * 0.2
        end_x, end_y = size * 0.8, size * 0.8
        
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t
            
            # Add curvature and randomness
            curve_offset = math.sin(t * math.pi) * size * 0.1
            x += curve_offset + random.uniform(-1, 1)
            y += random.uniform(-1, 1)
            
            points.append((int(x), int(y)))
        
        return points
    
    def _create_number_stroke(self, char, size):
        """Create stroke pattern for numbers"""
        # Simplified number patterns
        patterns = {
            '0': [(0, 0), (size, 0), (size, size), (0, size), (0, 0)],
            '1': [(size//2, 0), (size//2, size)],
            '2': [(0, 0), (size, 0), (size, size//2), (0, size), (size, size)],
            # Add more patterns as needed
        }
        
        pattern = patterns.get(char, [(0, 0), (size//2, size//2)])
        
        # Add slight randomness
        randomized_pattern = []
        for x, y in pattern:
            x += random.uniform(-2, 2)
            y += random.uniform(-2, 2)
            randomized_pattern.append((int(x), int(y)))
        
        return randomized_pattern
    
    def _create_punctuation_stroke(self, char, size):
        """Create stroke pattern for punctuation"""
        patterns = {
            '.': [(size//2, size * 0.8), (size//2, size * 0.9)],
            ',': [(size//2, size * 0.8), (size//3, size * 0.9)],
            '!': [(size//2, 0), (size//2, size * 0.7), (size//2, size * 0.8), (size//2, size * 0.9)],
            '?': [(0, size * 0.2), (size * 0.8, 0), (size * 0.8, size * 0.4), (size//2, size * 0.6)],
        }
        
        return patterns.get(char, [(0, 0), (size//2, size//2)])
    
    def _get_character_variations(self):
        """Get random variations for character"""
        return {
            'rotation': random.uniform(*self.char_variations['rotation_range']),
            'scale': random.uniform(*self.char_variations['scale_range']),
            'offset': (
                random.randint(*self.char_variations['offset_range']),
                random.randint(*self.char_variations['offset_range'])
            )
        }
    
    def _get_character_width(self, char, font_size):
        """Get approximate character width"""
        # Simplified character width estimation
        if char == ' ':
            return font_size * 0.3
        elif char in 'iIl':
            return font_size * 0.3
        elif char in 'mMwW':
            return font_size * 0.8
        else:
            return font_size * 0.6
    
    def _rotate_strokes(self, strokes, angle):
        """Rotate stroke patterns"""
        rotated_strokes = []
        angle_rad = math.radians(angle)
        
        for stroke in strokes:
            rotated_stroke = []
            for x, y in stroke:
                # Rotate around origin
                new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
                new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
                rotated_stroke.append((int(new_x), int(new_y)))
            rotated_strokes.append(rotated_stroke)
        
        return rotated_strokes
    
    def _scale_strokes(self, strokes, scale):
        """Scale stroke patterns"""
        scaled_strokes = []
        
        for stroke in strokes:
            scaled_stroke = []
            for x, y in stroke:
                scaled_stroke.append((int(x * scale), int(y * scale)))
            scaled_strokes.append(scaled_stroke)
        
        return scaled_strokes
    
    def _add_paper_effects(self, image):
        """Add paper texture and aging effects"""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Add subtle paper grain
            noise = np.random.normal(0, 2, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            
            # Add occasional ink spots
            height, width = img_array.shape[:2]
            for _ in range(random.randint(1, 5)):
                if random.random() < 0.3:  # 30% chance
                    spot_x = random.randint(0, width - 1)
                    spot_y = random.randint(0, height - 1)
                    spot_size = random.randint(1, 3)
                    
                    cv2.circle(img_array, (spot_x, spot_y), spot_size, (30, 30, 60), -1)
            
            return Image.fromarray(img_array.astype(np.uint8))
        
        except Exception as e:
            print(f"Error adding paper effects: {str(e)}")
            return image
    
    def get_available_styles(self):
        """Get list of available handwriting styles"""
        return self.handwriting_styles.get_available_styles()
    
    def preview_style(self, style_name, sample_text="Sample Text"):
        """Generate a preview of the handwriting style"""
        try:
            return self.generate_handwriting(
                text=sample_text,
                style=style_name,
                font_size=24,
                add_noise=True
            )
        except Exception as e:
            print(f"Error generating style preview: {str(e)}")
            return None
