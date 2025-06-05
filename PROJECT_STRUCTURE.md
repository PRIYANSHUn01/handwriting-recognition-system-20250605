# Project Structure

```
handwriting-recognition-generation/
├── app.py                          # Main Streamlit application
├── README.md                       # Project documentation
├── CHANGELOG.md                    # Version history
├── DEPLOYMENT.md                   # Deployment instructions
├── LICENSE                         # MIT License
├── setup.py                        # Package configuration
├── .gitignore                      # Git ignore rules
├── pyproject.toml                  # Project dependencies
├── uv.lock                         # Dependency lock file
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── database/
│   ├── models.py                   # SQLAlchemy database models
│   └── service.py                  # Database service layer
│
├── models/
│   ├── cnn_model.py               # TensorFlow/PyTorch CNN implementation
│   ├── simple_cnn.py              # NumPy-based CNN for stability
│   └── style_model.py             # Custom style training model
│
├── utils/
│   ├── handwriting_generator.py   # Text-to-handwriting conversion
│   ├── image_processor.py         # Image preprocessing utilities
│   ├── ocr_engine.py              # OCR processing engine
│   └── github_integration.py      # GitHub API integration
│
├── data/
│   └── fonts/
│       └── handwriting_styles.py  # Built-in handwriting styles
│
└── saved_styles/                  # Directory for custom styles
```

## Component Overview

### Core Application
- **app.py**: Main Streamlit interface with all tabs and functionality
- **pyproject.toml**: Project configuration and dependencies
- **.streamlit/config.toml**: Streamlit server configuration

### Database Layer
- **models.py**: Complete SQLAlchemy schema with 6 core tables
- **service.py**: Database service layer with CRUD operations

### Machine Learning Models
- **cnn_model.py**: Advanced CNN with TensorFlow/PyTorch support
- **simple_cnn.py**: Stable NumPy-based CNN implementation
- **style_model.py**: Custom handwriting style training system

### Utility Modules
- **handwriting_generator.py**: 9 built-in styles and text conversion
- **image_processor.py**: Advanced image preprocessing pipeline
- **ocr_engine.py**: Multi-language OCR with confidence scoring
- **github_integration.py**: Complete GitHub API integration

### Data Management
- **handwriting_styles.py**: Built-in style configurations
- **saved_styles/**: Directory for user-created custom styles

## Key Features

### Text Recognition
- Multi-language OCR support (8 languages)
- Advanced image preprocessing
- Confidence scoring and validation
- Batch processing capabilities

### Handwriting Generation
- 9 professionally designed styles
- Custom style training
- Parameter customization
- Multiple output formats

### Neural Network AI
- CNN-based character recognition
- Style analysis metrics
- Model training and inference
- Performance optimization

### GitHub Integration
- Repository management
- Version control for styles
- Collaborative features
- Automated backups

### Database Features
- User session management
- Complete history tracking
- Real-time analytics
- Performance monitoring

## Architecture Principles

### Modular Design
- Separation of concerns
- Reusable components
- Clean interfaces
- Scalable structure

### Data Flow
1. User input through Streamlit interface
2. Processing through utility modules
3. ML model inference and training
4. Database persistence
5. GitHub version control
6. Results presentation

### Error Handling
- Comprehensive exception handling
- Graceful degradation
- User-friendly error messages
- Logging and monitoring

### Performance Optimization
- Cached model loading
- Efficient database queries
- Optimized image processing
- Responsive user interface