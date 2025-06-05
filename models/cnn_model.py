try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
import numpy as np
import cv2
from PIL import Image
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class HandwritingCNN:
    """CNN-based handwriting recognition and style analysis model"""
    
    def __init__(self, model_type='tensorflow'):
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.input_shape = (64, 256, 1)  # Height, Width, Channels
        self.model_path = 'models/saved_cnn_model'
        self.char_classes = self._get_character_classes()
        
    def _get_character_classes(self):
        """Define character classes for recognition"""
        chars = []
        # Letters
        chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        # Numbers
        chars.extend([str(i) for i in range(10)])
        # Common punctuation
        chars.extend([' ', '.', ',', '!', '?', "'", '"', '-', '(', ')', ':'])
        return chars
    
    def build_tensorflow_model(self):
        """Build CNN model using TensorFlow/Keras"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(len(self.char_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_pytorch_model(self):
        """Build CNN model using PyTorch"""
        class HandwritingNet(nn.Module):
            def __init__(self, num_classes):
                super(HandwritingNet, self).__init__()
                
                # Convolutional layers
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
                self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                
                # Batch normalization
                self.bn1 = nn.BatchNorm2d(32)
                self.bn2 = nn.BatchNorm2d(32)
                self.bn3 = nn.BatchNorm2d(64)
                self.bn4 = nn.BatchNorm2d(64)
                self.bn5 = nn.BatchNorm2d(128)
                self.bn6 = nn.BatchNorm2d(128)
                self.bn7 = nn.BatchNorm2d(256)
                self.bn8 = nn.BatchNorm2d(256)
                
                # Pooling and dropout
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout_conv = nn.Dropout2d(0.25)
                self.dropout_fc = nn.Dropout(0.5)
                
                # Calculate the size after convolutions
                self.fc1_input_size = self._get_conv_output_size()
                
                # Fully connected layers
                self.fc1 = nn.Linear(self.fc1_input_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, num_classes)
                
                self.bn_fc1 = nn.BatchNorm1d(512)
                self.bn_fc2 = nn.BatchNorm1d(256)
                
            def _get_conv_output_size(self):
                # Calculate output size after all conv and pooling layers
                h, w = 64, 256  # Input dimensions
                # After 4 pooling operations (each divides by 2)
                h = h // (2**4)  # 4
                w = w // (2**4)  # 16
                return 256 * h * w  # 256 channels * 4 * 16
                
            def forward(self, x):
                # First conv block
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                x = self.dropout_conv(x)
                
                # Second conv block
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.relu(self.bn4(self.conv4(x)))
                x = self.pool(x)
                x = self.dropout_conv(x)
                
                # Third conv block
                x = F.relu(self.bn5(self.conv5(x)))
                x = F.relu(self.bn6(self.conv6(x)))
                x = self.pool(x)
                x = self.dropout_conv(x)
                
                # Fourth conv block
                x = F.relu(self.bn7(self.conv7(x)))
                x = F.relu(self.bn8(self.conv8(x)))
                x = self.pool(x)
                x = self.dropout_conv(x)
                
                # Fully connected layers
                x = x.view(x.size(0), -1)
                x = F.relu(self.bn_fc1(self.fc1(x)))
                x = self.dropout_fc(x)
                x = F.relu(self.bn_fc2(self.fc2(x)))
                x = self.dropout_fc(x)
                x = self.fc3(x)
                
                return F.log_softmax(x, dim=1)
        
        return HandwritingNet(len(self.char_classes))
    
    def initialize_model(self):
        """Initialize the CNN model"""
        try:
            if self.model_type == 'tensorflow':
                self.model = self.build_tensorflow_model()
                print(f"TensorFlow CNN model initialized with {self.model.count_params()} parameters")
            elif self.model_type == 'pytorch':
                self.model = self.build_pytorch_model()
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"PyTorch CNN model initialized with {total_params} parameters")
            
            # Print model architecture
            if self.model_type == 'tensorflow':
                self.model.summary()
            else:
                print(self.model)
                
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            return False
        
        return True
    
    def preprocess_image(self, image):
        """Preprocess image for CNN input"""
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Resize to model input size
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Threshold to binary
            _, image = cv2.threshold(image, 0.5, 1.0, cv2.THRESH_BINARY)
            
            # Invert if needed (text should be white on black background)
            if np.mean(image) > 0.5:
                image = 1.0 - image
            
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic handwriting data for training"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random
            
            images = []
            labels = []
            
            # Create synthetic handwriting samples
            for _ in range(num_samples):
                # Random character
                char = random.choice(self.char_classes)
                
                # Create image
                img = Image.new('RGB', (self.input_shape[1], self.input_shape[0]), 'white')
                draw = ImageDraw.Draw(img)
                
                # Try to use a font, fallback to default
                try:
                    font_size = random.randint(24, 48)
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Random position and style variations
                x = random.randint(10, self.input_shape[1] - 50)
                y = random.randint(5, self.input_shape[0] - 30)
                
                # Draw character with variations
                draw.text((x, y), char, fill='black', font=font)
                
                # Add noise and variations
                if random.random() > 0.7:
                    # Add slight rotation
                    angle = random.randint(-15, 15)
                    img = img.rotate(angle, fillcolor='white')
                
                # Preprocess
                processed_img = self.preprocess_image(img)
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(char)
            
            return np.array(images), np.array(labels)
            
        except Exception as e:
            print(f"Error generating synthetic data: {str(e)}")
            return None, None
    
    def train_model(self, training_images=None, training_labels=None, epochs=20, batch_size=32):
        """Train the CNN model"""
        try:
            if self.model is None:
                print("Model not initialized. Call initialize_model() first.")
                return False
            
            # Generate training data if not provided
            if training_images is None or training_labels is None:
                print("Generating synthetic training data...")
                training_images, training_labels = self.generate_synthetic_data(5000)
                
                if training_images is None:
                    print("Failed to generate training data")
                    return False
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(training_labels)
            
            if self.model_type == 'tensorflow':
                # Convert to categorical for TensorFlow
                categorical_labels = keras.utils.to_categorical(encoded_labels, len(self.char_classes))
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    training_images, categorical_labels, test_size=0.2, random_state=42
                )
                
                # Callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                    keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True)
                ]
                
                # Train model
                history = self.model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Plot training history
                self._plot_training_history(history)
                
            elif self.model_type == 'pytorch':
                # PyTorch training implementation
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                
                # Convert to PyTorch tensors
                X_tensor = torch.FloatTensor(training_images).permute(0, 3, 1, 2)  # NHWC to NCHW
                y_tensor = torch.LongTensor(encoded_labels)
                
                # Split data
                dataset = TensorDataset(X_tensor, y_tensor)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                
                # Optimizer and loss
                optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                # Training loop
                train_losses = []
                val_accuracies = []
                
                for epoch in range(epochs):
                    # Training
                    self.model.train()
                    train_loss = 0
                    for batch_idx, (data, target) in enumerate(train_loader):
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # Validation
                    self.model.eval()
                    val_loss = 0
                    correct = 0
                    with torch.no_grad():
                        for data, target in val_loader:
                            output = self.model(data)
                            val_loss += criterion(output, target).item()
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)).sum().item()
                    
                    val_accuracy = correct / len(val_dataset)
                    train_losses.append(train_loss / len(train_loader))
                    val_accuracies.append(val_accuracy)
                    
                    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
                
                # Save model
                torch.save(self.model.state_dict(), self.model_path + '_pytorch.pth')
            
            print("Model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def _plot_training_history(self, history):
        """Plot training history for TensorFlow model"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot training & validation accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot training & validation loss
            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('models/training_history.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")
    
    def predict_character(self, image):
        """Predict character from preprocessed image"""
        try:
            if self.model is None:
                return None, 0.0
            
            # Preprocess image
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return None, 0.0
            
            # Add batch dimension
            input_img = np.expand_dims(processed_img, axis=0)
            
            if self.model_type == 'tensorflow':
                # TensorFlow prediction
                predictions = self.model.predict(input_img, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]))
                
            elif self.model_type == 'pytorch':
                # PyTorch prediction
                self.model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_img).permute(0, 3, 1, 2)
                    output = self.model(input_tensor)
                    probabilities = torch.exp(output)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = float(torch.max(probabilities))
            
            # Decode prediction
            predicted_char = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return predicted_char, confidence
            
        except Exception as e:
            print(f"Error predicting character: {str(e)}")
            return None, 0.0
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        try:
            if filepath is None:
                filepath = self.model_path
            
            if self.model_type == 'tensorflow':
                self.model.save(filepath)
                # Also save label encoder
                with open(filepath + '_labels.json', 'w') as f:
                    json.dump({
                        'classes': self.label_encoder.classes_.tolist(),
                        'char_classes': self.char_classes
                    }, f)
                    
            elif self.model_type == 'pytorch':
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'label_encoder': self.label_encoder,
                    'char_classes': self.char_classes,
                    'input_shape': self.input_shape
                }, filepath + '_pytorch.pth')
            
            print(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        try:
            if filepath is None:
                filepath = self.model_path
            
            if self.model_type == 'tensorflow':
                if os.path.exists(filepath):
                    self.model = keras.models.load_model(filepath)
                    
                    # Load label encoder
                    labels_file = filepath + '_labels.json'
                    if os.path.exists(labels_file):
                        with open(labels_file, 'r') as f:
                            data = json.load(f)
                            self.label_encoder.classes_ = np.array(data['classes'])
                            self.char_classes = data['char_classes']
                    
                    print(f"TensorFlow model loaded from {filepath}")
                    return True
                    
            elif self.model_type == 'pytorch':
                filepath_pytorch = filepath + '_pytorch.pth'
                if os.path.exists(filepath_pytorch):
                    checkpoint = torch.load(filepath_pytorch, map_location='cpu')
                    
                    # Initialize model
                    self.model = self.build_pytorch_model()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.label_encoder = checkpoint['label_encoder']
                    self.char_classes = checkpoint['char_classes']
                    self.input_shape = checkpoint['input_shape']
                    
                    print(f"PyTorch model loaded from {filepath_pytorch}")
                    return True
            
            print(f"Model file not found: {filepath}")
            return False
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def analyze_handwriting_style(self, image):
        """Analyze handwriting style characteristics using CNN features"""
        try:
            if self.model is None:
                return None
            
            # Preprocess image
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return None
            
            # Extract features from intermediate layers
            if self.model_type == 'tensorflow':
                # Create feature extraction model
                feature_extractor = keras.Model(
                    inputs=self.model.input,
                    outputs=[layer.output for layer in self.model.layers[:-3]]  # Skip final dense layers
                )
                
                input_img = np.expand_dims(processed_img, axis=0)
                features = feature_extractor.predict(input_img, verbose=0)
                
                # Analyze features to determine style characteristics
                conv_features = features[-1]  # Last convolutional features
                
                # Calculate style metrics
                style_metrics = {
                    'texture_complexity': float(np.std(conv_features)),
                    'edge_density': float(np.mean(np.abs(conv_features))),
                    'stroke_variation': float(np.var(conv_features)),
                    'writing_consistency': float(1.0 / (1.0 + np.std(conv_features))),
                    'slant_estimate': self._estimate_slant(processed_img),
                    'thickness_variation': self._estimate_thickness_variation(processed_img)
                }
                
                return style_metrics
                
        except Exception as e:
            print(f"Error analyzing handwriting style: {str(e)}")
            return None
    
    def _estimate_slant(self, image):
        """Estimate handwriting slant angle"""
        try:
            # Use Hough line detection to estimate slant
            edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90  # Convert to slant angle
                    if -45 <= angle <= 45:  # Valid slant range
                        angles.append(angle)
                
                if angles:
                    return float(np.median(angles))
            
            return 0.0  # No slant detected
            
        except:
            return 0.0
    
    def _estimate_thickness_variation(self, image):
        """Estimate stroke thickness variation"""
        try:
            # Calculate thickness variation using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Erosion and dilation to measure thickness
            eroded = cv2.erode((image * 255).astype(np.uint8), kernel, iterations=1)
            dilated = cv2.dilate((image * 255).astype(np.uint8), kernel, iterations=1)
            
            thickness_map = dilated - eroded
            thickness_variation = float(np.std(thickness_map[thickness_map > 0]))
            
            return thickness_variation
            
        except:
            return 0.0
    
    def get_model_info(self):
        """Get information about the current model"""
        info = {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'num_classes': len(self.char_classes),
            'character_classes': self.char_classes,
            'model_initialized': self.model is not None
        }
        
        if self.model is not None:
            if self.model_type == 'tensorflow':
                info['total_parameters'] = self.model.count_params()
            elif self.model_type == 'pytorch':
                info['total_parameters'] = sum(p.numel() for p in self.model.parameters())
        
        return info