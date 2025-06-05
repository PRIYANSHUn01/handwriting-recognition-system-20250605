class HandwritingStyles:
    """Manages different handwriting styles and their configurations"""
    
    def __init__(self):
        self.styles = {
            'casual': {
                'name': 'Casual',
                'description': 'Relaxed, everyday handwriting style',
                'color': (25, 25, 60),
                'slant_angle': 2,
                'letter_spacing': 1.1,
                'line_spacing': 1.4,
                'stroke_thickness': 2,
                'character_variations': {
                    'size': 0.15,
                    'rotation': 3,
                    'spacing': 0.2
                },
                'noise_level': 0.12
            },
            
            'formal': {
                'name': 'Formal',
                'description': 'Clean, professional handwriting style',
                'color': (15, 15, 45),
                'slant_angle': 0,
                'letter_spacing': 1.0,
                'line_spacing': 1.6,
                'stroke_thickness': 1,
                'character_variations': {
                    'size': 0.08,
                    'rotation': 1,
                    'spacing': 0.1
                },
                'noise_level': 0.05
            },
            
            'cursive': {
                'name': 'Cursive',
                'description': 'Flowing, connected handwriting style',
                'color': (30, 20, 70),
                'slant_angle': 15,
                'letter_spacing': 0.8,
                'line_spacing': 1.5,
                'stroke_thickness': 2,
                'character_variations': {
                    'size': 0.12,
                    'rotation': 5,
                    'spacing': 0.15
                },
                'noise_level': 0.1,
                'connected': True
            },
            
            'bold': {
                'name': 'Bold',
                'description': 'Strong, heavy handwriting style',
                'color': (10, 10, 40),
                'slant_angle': 0,
                'letter_spacing': 1.2,
                'line_spacing': 1.7,
                'stroke_thickness': 4,
                'character_variations': {
                    'size': 0.1,
                    'rotation': 2,
                    'spacing': 0.15
                },
                'noise_level': 0.08
            },
            
            'artistic': {
                'name': 'Artistic',
                'description': 'Creative, expressive handwriting style',
                'color': (35, 15, 55),
                'slant_angle': -5,
                'letter_spacing': 1.3,
                'line_spacing': 1.8,
                'stroke_thickness': 3,
                'character_variations': {
                    'size': 0.25,
                    'rotation': 8,
                    'spacing': 0.3
                },
                'noise_level': 0.18
            },
            
            'neat': {
                'name': 'Neat',
                'description': 'Precise, carefully written style',
                'color': (20, 20, 50),
                'slant_angle': 1,
                'letter_spacing': 1.0,
                'line_spacing': 1.5,
                'stroke_thickness': 1,
                'character_variations': {
                    'size': 0.05,
                    'rotation': 0.5,
                    'spacing': 0.05
                },
                'noise_level': 0.03
            },
            
            'vintage': {
                'name': 'Vintage',
                'description': 'Classic, old-fashioned fountain pen style',
                'color': (45, 35, 25),  # Brown ink
                'slant_angle': 3,
                'letter_spacing': 1.1,
                'line_spacing': 1.6,
                'stroke_thickness': 2,
                'character_variations': {
                    'size': 0.15,
                    'rotation': 4,
                    'spacing': 0.2
                },
                'noise_level': 0.15,
                'ink_blots': True
            },
            
            'quick': {
                'name': 'Quick',
                'description': 'Fast, hurried handwriting style',
                'color': (25, 25, 65),
                'slant_angle': 8,
                'letter_spacing': 0.9,
                'line_spacing': 1.3,
                'stroke_thickness': 2,
                'character_variations': {
                    'size': 0.2,
                    'rotation': 6,
                    'spacing': 0.25
                },
                'noise_level': 0.16
            }
        }
        
        # Character-specific adjustments
        self.character_adjustments = {
            'vowels': ['a', 'e', 'i', 'o', 'u'],
            'tall_letters': ['b', 'd', 'f', 'h', 'k', 'l', 't'],
            'descenders': ['g', 'j', 'p', 'q', 'y'],
            'wide_letters': ['m', 'w'],
            'narrow_letters': ['i', 'l', 'j']
        }
    
    def get_available_styles(self):
        """Get list of available style names"""
        return list(self.styles.keys())
    
    def get_style_config(self, style_name):
        """Get configuration for a specific style"""
        return self.styles.get(style_name, self.styles['casual'])
    
    def get_style_info(self, style_name):
        """Get detailed information about a style"""
        style = self.styles.get(style_name)
        if style:
            return {
                'name': style['name'],
                'description': style['description'],
                'characteristics': {
                    'slant': f"{style['slant_angle']}°",
                    'thickness': style['stroke_thickness'],
                    'spacing': style['letter_spacing'],
                    'variation': style['character_variations']['size']
                }
            }
        return None
    
    def get_all_styles_info(self):
        """Get information about all available styles"""
        styles_info = {}
        for style_key in self.styles:
            styles_info[style_key] = self.get_style_info(style_key)
        return styles_info
    
    def add_custom_style(self, style_name, config):
        """Add a custom style configuration"""
        try:
            # Validate required fields
            required_fields = ['color', 'slant_angle', 'letter_spacing', 'stroke_thickness']
            for field in required_fields:
                if field not in config:
                    return False
            
            # Set defaults for missing optional fields
            default_config = {
                'name': style_name.title(),
                'description': f'Custom style: {style_name}',
                'line_spacing': 1.5,
                'character_variations': {
                    'size': 0.1,
                    'rotation': 2,
                    'spacing': 0.1
                },
                'noise_level': 0.1
            }
            
            # Merge with provided config
            full_config = {**default_config, **config}
            
            # Add to styles
            self.styles[style_name] = full_config
            
            return True
        
        except Exception as e:
            print(f"Error adding custom style: {str(e)}")
            return False
    
    def remove_style(self, style_name):
        """Remove a style (only custom styles)"""
        default_styles = ['casual', 'formal', 'cursive', 'bold', 'artistic', 'neat', 'vintage', 'quick']
        
        if style_name not in default_styles and style_name in self.styles:
            del self.styles[style_name]
            return True
        
        return False
    
    def get_character_adjustments(self, character):
        """Get specific adjustments for a character"""
        adjustments = {
            'height_multiplier': 1.0,
            'width_multiplier': 1.0,
            'baseline_offset': 0
        }
        
        char_lower = character.lower()
        
        # Adjust for tall letters
        if char_lower in self.character_adjustments['tall_letters']:
            adjustments['height_multiplier'] = 1.3
            adjustments['baseline_offset'] = -0.2
        
        # Adjust for descenders
        elif char_lower in self.character_adjustments['descenders']:
            adjustments['height_multiplier'] = 1.2
            adjustments['baseline_offset'] = 0.3
        
        # Adjust for wide letters
        if char_lower in self.character_adjustments['wide_letters']:
            adjustments['width_multiplier'] = 1.4
        
        # Adjust for narrow letters
        elif char_lower in self.character_adjustments['narrow_letters']:
            adjustments['width_multiplier'] = 0.6
        
        return adjustments
    
    def validate_style_config(self, config):
        """Validate a style configuration"""
        try:
            required_fields = {
                'color': tuple,
                'slant_angle': (int, float),
                'letter_spacing': (int, float),
                'stroke_thickness': int
            }
            
            for field, expected_type in required_fields.items():
                if field not in config:
                    return False, f"Missing required field: {field}"
                
                if not isinstance(config[field], expected_type):
                    return False, f"Invalid type for {field}: expected {expected_type}"
            
            # Validate ranges
            if not (-30 <= config['slant_angle'] <= 30):
                return False, "Slant angle must be between -30 and 30 degrees"
            
            if not (0.5 <= config['letter_spacing'] <= 3.0):
                return False, "Letter spacing must be between 0.5 and 3.0"
            
            if not (1 <= config['stroke_thickness'] <= 10):
                return False, "Stroke thickness must be between 1 and 10"
            
            if len(config['color']) != 3 or not all(0 <= c <= 255 for c in config['color']):
                return False, "Color must be RGB tuple with values 0-255"
            
            return True, "Valid configuration"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def create_style_preview_text(self, style_name):
        """Create sample text that showcases the style"""
        style = self.styles.get(style_name, self.styles['casual'])
        
        # Choose preview text based on style characteristics
        if style.get('connected'):
            preview_text = "Beautiful flowing script with connected letters"
        elif style['character_variations']['size'] > 0.2:
            preview_text = "Expressive and artistic handwriting style"
        elif style['stroke_thickness'] >= 4:
            preview_text = "Bold and strong lettering"
        elif style['noise_level'] < 0.05:
            preview_text = "Precise and neat handwriting"
        else:
            preview_text = f"Sample text in {style['name']} handwriting style"
        
        return preview_text
    
    def get_style_recommendations(self, use_case):
        """Recommend styles based on use case"""
        recommendations = {
            'note_taking': ['quick', 'casual', 'neat'],
            'formal_documents': ['formal', 'neat', 'cursive'],
            'creative_writing': ['artistic', 'vintage', 'cursive'],
            'signatures': ['cursive', 'artistic', 'formal'],
            'practice': ['casual', 'formal', 'neat'],
            'decorative': ['artistic', 'vintage', 'bold']
        }
        
        return recommendations.get(use_case, ['casual', 'formal', 'neat'])
