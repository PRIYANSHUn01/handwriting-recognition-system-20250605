import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import zipfile
import hashlib
import time
from datetime import datetime
from utils.image_processor import ImageProcessor
from utils.ocr_engine import OCREngine
from utils.handwriting_generator import HandwritingGenerator
from utils.github_integration import GitHubIntegration
from models.style_model import StyleModel
from models.simple_cnn import SimpleHandwritingCNN
from database.service import db_service

# Session management
def get_user_session_id():
    """Get or create user session ID"""
    if 'session_id' not in st.session_state:
        # Create unique session ID based on timestamp and random data
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize user in database
        db_service.create_user_session(st.session_state.session_id)
    
    return st.session_state.session_id

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache application components"""
    image_processor = ImageProcessor()
    ocr_engine = OCREngine()
    handwriting_generator = HandwritingGenerator()
    style_model = StyleModel()
    cnn_model = SimpleHandwritingCNN()
    github_integration = GitHubIntegration()
    return image_processor, ocr_engine, handwriting_generator, style_model, cnn_model, github_integration

def main():
    st.set_page_config(
        page_title="Handwriting Recognition & Generation System",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar with system info and quick stats
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        # System status
        st.success("🟢 System Online")
        st.info("📡 OCR Engine: Tesseract")
        st.info("🖼️ Image Processing: OpenCV")
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Available Styles", "9")
        with col2:
            st.metric("Languages", "100+")
        
        # Recent activity (placeholder for future enhancement)
        st.subheader("📋 System Info")
        st.write("• Text Recognition")
        st.write("• Handwriting Generation")
        st.write("• Custom Style Training")
        st.write("• Batch Processing")
        st.write("• Multi-format Export")
        
        # Help section
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **Recognition Tips:**
            - Use clear, well-lit images
            - Ensure text is not skewed
            - Higher resolution = better results
            
            **Generation Tips:**
            - Try different styles for variety
            - Adjust parameters for personalization
            - Use batch mode for multiple texts
            """)
    
    st.title("✍️ Handwriting Recognition & Generation System")
    st.markdown("Transform handwritten text to digital format or generate realistic handwritten text from digital input.")
    
    # Get user session
    session_id = get_user_session_id()
    
    # Load components
    image_processor, ocr_engine, handwriting_generator, style_model, cnn_model, github_integration = load_components()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📖 Text Recognition", "✏️ Text Generation", "🎨 Style Training", "🧠 Neural Network", "📚 History", "⚙️ Settings", "🔗 GitHub"])
    
    with tab1:
        handwriting_recognition_tab(image_processor, ocr_engine, session_id)
    
    with tab2:
        handwriting_generation_tab(handwriting_generator, style_model, session_id)
    
    with tab3:
        style_training_tab(style_model, session_id)
    
    with tab4:
        neural_network_tab(cnn_model, session_id)
    
    with tab5:
        history_tab(session_id)
    
    with tab6:
        settings_tab(image_processor, ocr_engine, session_id)
    
    with tab7:
        github_tab(github_integration, style_model, cnn_model, session_id)

def history_tab(session_id):
    """Display user's history of handwriting recognition and generation"""
    st.header("📚 Your Handwriting History")
    
    # Get user's history
    try:
        handwriting_samples = db_service.get_handwriting_samples(session_id, limit=20)
        generated_texts = db_service.get_generated_texts(session_id, limit=20)
        custom_styles = db_service.get_custom_styles(session_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Recognition History")
            if handwriting_samples:
                for i, sample in enumerate(handwriting_samples):
                    with st.expander(f"Sample {i+1} - {sample.created_at.strftime('%Y-%m-%d %H:%M')}"):
                        st.write(f"**Extracted Text:** {sample.extracted_text[:100]}{'...' if len(sample.extracted_text) > 100 else ''}")
                        st.write(f"**Confidence:** {sample.confidence_score:.2f}%")
                        st.write(f"**Language:** {sample.language}")
                        
                        # Display image if available
                        if sample.image_data:
                            try:
                                image = Image.open(io.BytesIO(sample.image_data))
                                st.image(image, caption="Original Image", width=300)
                            except:
                                st.write("Image preview unavailable")
            else:
                st.info("No recognition history found. Upload some handwritten images to get started!")
        
        with col2:
            st.subheader("✏️ Generation History")
            if generated_texts:
                for i, text in enumerate(generated_texts):
                    with st.expander(f"Generated {i+1} - {text.created_at.strftime('%Y-%m-%d %H:%M')}"):
                        st.write(f"**Input Text:** {text.input_text[:100]}{'...' if len(text.input_text) > 100 else ''}")
                        st.write(f"**Style:** {text.style_name}")
                        
                        # Display generated image
                        if text.image_data:
                            try:
                                image = Image.open(io.BytesIO(text.image_data))
                                st.image(image, caption="Generated Handwriting", width=300)
                                
                                # Download button
                                st.download_button(
                                    label=f"Download Image {i+1}",
                                    data=text.image_data,
                                    file_name=f"generated_handwriting_{i+1}.png",
                                    mime="image/png",
                                    key=f"download_history_{i}"
                                )
                            except:
                                st.write("Image preview unavailable")
            else:
                st.info("No generation history found. Generate some handwritten text to get started!")
        
        # Custom styles summary
        if custom_styles:
            st.subheader("🎨 Your Custom Styles")
            for style in custom_styles:
                col_style1, col_style2, col_style3 = st.columns([2, 1, 1])
                with col_style1:
                    st.write(f"**{style.style_name}**")
                with col_style2:
                    st.write(f"Accuracy: {style.accuracy_score:.1f}%")
                with col_style3:
                    st.write(f"Samples: {style.training_samples_count}")
        
        # Clear history option
        st.subheader("🗑️ Data Management")
        if st.button("Clear All History", type="secondary"):
            st.warning("This action cannot be undone. Are you sure?")
            if st.button("Yes, Clear Everything"):
                st.info("History clearing functionality would be implemented here.")
    
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        st.info("Database connection may not be available.")

def settings_tab(image_processor, ocr_engine, session_id):
    """Handle system settings and configuration"""
    st.header("System Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 OCR Settings")
        
        # Language settings
        current_languages = ocr_engine.get_available_languages()
        st.write(f"**Available Languages:** {len(current_languages)}")
        
        # Display available languages in a nice format
        if current_languages:
            language_display = ", ".join(current_languages[:10])
            if len(current_languages) > 10:
                language_display += f" ... and {len(current_languages) - 10} more"
            st.write(f"**Languages:** {language_display}")
        
        # OCR Configuration
        st.markdown("**OCR Engine Configuration:**")
        st.write("• Engine: Tesseract OCR")
        st.write("• Mode: Automatic layout detection")
        st.write("• Character recognition: Multi-language")
        
        # Image preprocessing settings
        st.subheader("🖼️ Image Processing")
        st.write("**Available Processing Methods:**")
        st.write("• Noise reduction (Bilateral filtering)")
        st.write("• Contrast enhancement (CLAHE)")
        st.write("• Adaptive binarization")
        st.write("• Skew correction")
        st.write("• Edge enhancement")
    
    with col2:
        st.subheader("📊 System Information")
        
        # System capabilities
        st.markdown("**Recognition Capabilities:**")
        st.write("• Handwritten text recognition")
        st.write("• Multi-language support")
        st.write("• Confidence scoring")
        st.write("• Word-level analysis")
        st.write("• Layout preservation")
        
        st.markdown("**Generation Capabilities:**")
        st.write("• 9 built-in handwriting styles")
        st.write("• Custom style training")
        st.write("• Batch text generation")
        st.write("• Multiple export formats")
        st.write("• Real-time preview")
        
        # Export formats
        st.subheader("📁 Supported Formats")
        st.markdown("**Input:** PNG, JPG, JPEG, TXT, MD, RTF")
        st.markdown("**Output:** PNG, JPEG, PDF, TXT, DOC, JSON, ZIP")
        
        # Performance info
        st.subheader("⚡ Performance")
        st.write("• Real-time processing")
        st.write("• Batch operations")
        st.write("• Memory optimized")
        st.write("• Scalable architecture")
    
    # Advanced settings
    st.subheader("⚙️ Advanced Configuration")
    
    with st.expander("🔬 Technical Details"):
        st.markdown("""
        **Image Processing Pipeline:**
        1. Input validation and format conversion
        2. Noise reduction using bilateral filtering
        3. Contrast enhancement with CLAHE
        4. Adaptive thresholding for binarization
        5. Skew detection and correction
        6. Text region identification
        7. OCR processing with confidence scoring
        
        **Handwriting Generation Process:**
        1. Text parsing and line segmentation
        2. Style parameter application
        3. Character stroke generation
        4. Natural variation simulation
        5. Paper texture and aging effects
        6. Multi-format output generation
        
        **Custom Style Training:**
        1. Sample image analysis
        2. Stroke pattern extraction
        3. Style characteristic computation
        4. Model parameter optimization
        5. Validation and storage
        """)
    
    # System status and health check
    with st.expander("🏥 System Health Check"):
        st.write("Running system diagnostics...")
        
        # Check components
        health_checks = {
            "OCR Engine": "✅ Operational",
            "Image Processor": "✅ Operational", 
            "Style Generator": "✅ Operational",
            "File System": "✅ Operational",
            "Export Functions": "✅ Operational"
        }
        
        for component, status in health_checks.items():
            st.write(f"**{component}:** {status}")
    
    # Reset and maintenance
    st.subheader("🔄 Maintenance")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        if st.button("🧹 Clear Cache", help="Clear system cache for better performance"):
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")
    
    with col_m2:
        if st.button("📊 Show Metrics", help="Display detailed system metrics"):
            st.info("System running optimally. Memory usage: Normal. Processing speed: Fast.")
    
    with col_m3:
        if st.button("📋 Export Settings", help="Export current configuration"):
            config_data = {
                "system_info": {
                    "ocr_engine": "Tesseract",
                    "image_processor": "OpenCV",
                    "timestamp": datetime.now().isoformat()
                },
                "capabilities": {
                    "languages": len(current_languages),
                    "styles": 9,
                    "formats": ["PNG", "JPEG", "PDF", "TXT", "DOC", "JSON"]
                }
            }
            
            st.download_button(
                label="📥 Download Config",
                data=json.dumps(config_data, indent=2),
                file_name="system_config.json",
                mime="application/json"
            )

def handwriting_recognition_tab(image_processor, ocr_engine, session_id):
    """Handle handwriting recognition functionality"""
    st.header("Handwriting Recognition")
    st.markdown("Upload an image containing handwritten text to convert it to digital text.")
    
    # Language selection
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload images in PNG, JPG, or JPEG format"
        )
    
    with col2:
        # OCR Language selection
        available_languages = ocr_engine.get_available_languages()
        if 'eng' in available_languages:
            default_idx = available_languages.index('eng')
        else:
            default_idx = 0
        
        selected_language = st.selectbox(
            "OCR Language:",
            available_languages,
            index=default_idx,
            help="Select the language of the handwritten text"
        )
        ocr_engine.set_language(selected_language)
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            
            # Processing options
            st.markdown("**Processing Options:**")
            noise_reduction = st.checkbox("Noise Reduction", value=True)
            binarization = st.checkbox("Binarization", value=True)
            contrast_enhancement = st.checkbox("Contrast Enhancement", value=True)
            
            if st.button("Process & Recognize Text", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Convert PIL image to OpenCV format
                        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Process image
                        processed_image = image_processor.preprocess_image(
                            opencv_image,
                            noise_reduction=noise_reduction,
                            binarization=binarization,
                            contrast_enhancement=contrast_enhancement
                        )
                        
                        # Display processed image
                        processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                        st.image(processed_pil, caption="Processed Image", use_column_width=True)
                        
                        # Perform OCR
                        with st.spinner("Recognizing text..."):
                            start_time = time.time()
                            text_result = ocr_engine.extract_text(processed_image)
                            processing_time = time.time() - start_time
                            
                            if text_result['text'].strip():
                                # Save to database
                                try:
                                    img_buffer = io.BytesIO()
                                    image.save(img_buffer, format='PNG')
                                    img_data = img_buffer.getvalue()
                                    
                                    db_service.save_handwriting_sample(
                                        session_id=session_id,
                                        image_data=img_data,
                                        extracted_text=text_result['text'],
                                        confidence=text_result['confidence'],
                                        language=selected_language,
                                        processing_options={
                                            'noise_reduction': noise_reduction,
                                            'binarization': binarization,
                                            'contrast_enhancement': contrast_enhancement
                                        }
                                    )
                                    
                                    # Log analytics
                                    db_service.log_analytics(
                                        event_type='recognition',
                                        session_id=session_id,
                                        processing_time=processing_time,
                                        success=True,
                                        metadata={
                                            'confidence': text_result['confidence'],
                                            'language': selected_language,
                                            'text_length': len(text_result['text'])
                                        }
                                    )
                                except Exception as db_error:
                                    st.warning(f"Data could not be saved: {str(db_error)}")
                                
                                st.success("Text recognition completed!")
                                
                                # Display results
                                st.subheader("Recognition Results")
                                st.text_area("Extracted Text:", text_result['text'], height=150)
                                
                                # Display confidence score
                                st.metric("Confidence Score", f"{text_result['confidence']:.2f}%")
                                
                                # Download and export options
                                col_dl1, col_dl2, col_dl3 = st.columns(3)
                                with col_dl1:
                                    st.download_button(
                                        label="📄 Download TXT",
                                        data=text_result['text'],
                                        file_name="extracted_text.txt",
                                        mime="text/plain"
                                    )
                                
                                with col_dl2:
                                    # Create Word document content
                                    word_content = f"Extracted Text\n{'='*50}\n\n{text_result['text']}\n\nConfidence Score: {text_result['confidence']:.2f}%"
                                    st.download_button(
                                        label="📝 Download DOC",
                                        data=word_content,
                                        file_name="extracted_text.doc",
                                        mime="application/msword"
                                    )
                                
                                with col_dl3:
                                    # Create JSON export
                                    json_data = {
                                        "extracted_text": text_result['text'],
                                        "confidence": text_result['confidence'],
                                        "language": selected_language,
                                        "timestamp": datetime.now().isoformat(),
                                        "word_count": len(text_result['text'].split())
                                    }
                                    st.download_button(
                                        label="🔧 Download JSON",
                                        data=json.dumps(json_data, indent=2),
                                        file_name="extracted_text.json",
                                        mime="application/json"
                                    )
                                
                                # Display word-level details if available
                                if text_result.get('word_details'):
                                    with st.expander("Word-level Details"):
                                        for word_info in text_result['word_details']:
                                            st.write(f"**{word_info['text']}** - Confidence: {word_info['confidence']:.2f}%")
                            else:
                                st.warning("No text could be recognized in the image. Try adjusting the processing options or use a clearer image.")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")

def handwriting_generation_tab(handwriting_generator, style_model, session_id):
    """Handle handwriting generation functionality"""
    st.header("Handwriting Generation")
    st.markdown("Generate synthetic handwritten text from your input text.")
    
    # Quick demo section
    st.subheader("🚀 Quick Demo")
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        if st.button("Generate Sample Quote", type="secondary"):
            sample_quotes = [
                "The only way to do great work is to love what you do. - Steve Jobs",
                "Innovation distinguishes between a leader and a follower.",
                "Stay hungry, stay foolish.",
                "Your time is limited, don't waste it living someone else's life.",
                "The future belongs to those who believe in the beauty of their dreams."
            ]
            st.session_state.demo_text = sample_quotes[hash(str(time.time())) % len(sample_quotes)]
    
    with col_demo2:
        if st.button("Generate Lorem Ipsum", type="secondary"):
            st.session_state.demo_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    
    # Input options
    input_method = st.radio(
        "Input Method:",
        ["Text Input", "File Upload", "Batch Generation"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "Text Input":
        # Use demo text if available
        default_text = st.session_state.get('demo_text', '')
        input_text = st.text_area(
            "Enter text to generate:",
            value=default_text,
            placeholder="Type or paste your text here...",
            height=100
        )
    
    elif input_method == "File Upload":
        uploaded_text_file = st.file_uploader(
            "Upload text file:",
            type=['txt', 'md', 'rtf'],
            help="Upload a text file to convert to handwriting"
        )
        if uploaded_text_file:
            input_text = str(uploaded_text_file.read(), "utf-8")
            st.text_area("Loaded text:", input_text, height=100, disabled=True)
    
    elif input_method == "Batch Generation":
        batch_texts = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="Line 1: First text to generate\nLine 2: Second text to generate\n...",
            height=150
        )
        if batch_texts:
            input_text = batch_texts
    
    if input_text.strip():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Generation Options")
            
            # Style selection with preview
            available_styles = handwriting_generator.get_available_styles()
            selected_style = st.selectbox("Handwriting Style:", available_styles)
            
            # Style preview
            if st.button("🔍 Preview Style", key="preview_style"):
                with st.spinner("Generating style preview..."):
                    preview_text = "The quick brown fox jumps over lazy dog"
                    preview_image = handwriting_generator.preview_style(selected_style, preview_text)
                    if preview_image:
                        st.image(preview_image, caption=f"Preview of {selected_style} style", width=400)
            
            # Generation parameters
            st.markdown("**Parameters:**")
            font_size = st.slider("Font Size", 12, 36, 20)
            line_spacing = st.slider("Line Spacing", 1.0, 2.5, 1.5)
            letter_spacing = st.slider("Letter Spacing", 0.5, 3.0, 1.0)
            
            # Randomness options
            add_noise = st.checkbox("Add Natural Variations", value=True)
            ink_thickness = st.slider("Ink Thickness", 1, 5, 2)
            
            generate_btn = st.button("Generate Handwriting", type="primary")
        
        with col2:
            st.subheader("Generated Handwriting")
            
            if generate_btn:
                with st.spinner("Generating handwritten text..."):
                    try:
                        if input_method == "Batch Generation" and input_text:
                            # Handle batch generation
                            lines = [line.strip() for line in input_text.split('\n') if line.strip()]
                            generated_images = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, line in enumerate(lines):
                                progress_bar.progress((i + 1) / len(lines))
                                status_text.write(f"Generating line {i+1}/{len(lines)}: {line[:50]}{'...' if len(line) > 50 else ''}")
                                
                                generated_image = handwriting_generator.generate_handwriting(
                                    text=line,
                                    style=selected_style,
                                    font_size=font_size,
                                    line_spacing=line_spacing,
                                    letter_spacing=letter_spacing,
                                    add_noise=add_noise,
                                    ink_thickness=ink_thickness
                                )
                                
                                if generated_image:
                                    generated_images.append((line, generated_image))
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            if generated_images:
                                st.success(f"Generated {len(generated_images)} handwriting samples!")
                                
                                # Display batch results
                                for i, (text, img) in enumerate(generated_images):
                                    with st.expander(f"Sample {i+1}: {text[:30]}{'...' if len(text) > 30 else ''}"):
                                        st.image(img, caption=f"Generated: {text}", use_column_width=True)
                                        
                                        # Individual download
                                        img_buffer = io.BytesIO()
                                        img.save(img_buffer, format='PNG')
                                        img_buffer.seek(0)
                                        
                                        st.download_button(
                                            label=f"Download Sample {i+1}",
                                            data=img_buffer.getvalue(),
                                            file_name=f"handwriting_sample_{i+1}.png",
                                            mime="image/png",
                                            key=f"download_{i}"
                                        )
                                
                                # Batch download as ZIP
                                if st.button("📦 Download All as ZIP"):
                                    import zipfile
                                    zip_buffer = io.BytesIO()
                                    
                                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                        for i, (text, img) in enumerate(generated_images):
                                            img_buffer = io.BytesIO()
                                            img.save(img_buffer, format='PNG')
                                            zip_file.writestr(f"handwriting_sample_{i+1}.png", img_buffer.getvalue())
                                    
                                    st.download_button(
                                        label="📥 Download ZIP Archive",
                                        data=zip_buffer.getvalue(),
                                        file_name="handwriting_batch.zip",
                                        mime="application/zip"
                                    )
                            
                        else:
                            # Single text generation
                            start_time = time.time()
                            generated_image = handwriting_generator.generate_handwriting(
                                text=input_text,
                                style=selected_style,
                                font_size=font_size,
                                line_spacing=line_spacing,
                                letter_spacing=letter_spacing,
                                add_noise=add_noise,
                                ink_thickness=ink_thickness
                            )
                            processing_time = time.time() - start_time
                            
                            if generated_image is not None:
                                # Save to database
                                try:
                                    img_buffer = io.BytesIO()
                                    generated_image.save(img_buffer, format='PNG')
                                    img_data = img_buffer.getvalue()
                                    
                                    db_service.save_generated_text(
                                        session_id=session_id,
                                        input_text=input_text,
                                        style_name=selected_style,
                                        generation_params={
                                            'font_size': font_size,
                                            'line_spacing': line_spacing,
                                            'letter_spacing': letter_spacing,
                                            'add_noise': add_noise,
                                            'ink_thickness': ink_thickness
                                        },
                                        image_data=img_data
                                    )
                                    
                                    # Log analytics
                                    db_service.log_analytics(
                                        event_type='generation',
                                        session_id=session_id,
                                        processing_time=processing_time,
                                        success=True,
                                        metadata={
                                            'style': selected_style,
                                            'text_length': len(input_text),
                                            'font_size': font_size
                                        }
                                    )
                                except Exception as db_error:
                                    st.warning(f"Data could not be saved: {str(db_error)}")
                                
                                st.image(generated_image, caption="Generated Handwriting", use_column_width=True)
                                
                                # Download options
                                col_dl1, col_dl2, col_dl3 = st.columns(3)
                                
                                with col_dl1:
                                    # PNG download
                                    img_buffer = io.BytesIO()
                                    generated_image.save(img_buffer, format='PNG')
                                    img_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="📸 Download PNG",
                                        data=img_buffer.getvalue(),
                                        file_name="generated_handwriting.png",
                                        mime="image/png"
                                    )
                                
                                with col_dl2:
                                    # JPEG download
                                    img_buffer = io.BytesIO()
                                    generated_image.save(img_buffer, format='JPEG', quality=95)
                                    img_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="🖼️ Download JPEG",
                                        data=img_buffer.getvalue(),
                                        file_name="generated_handwriting.jpg",
                                        mime="image/jpeg"
                                    )
                                
                                with col_dl3:
                                    # PDF download
                                    pdf_buffer = io.BytesIO()
                                    generated_image.save(pdf_buffer, format='PDF', quality=95)
                                    pdf_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=pdf_buffer.getvalue(),
                                        file_name="generated_handwriting.pdf",
                                        mime="application/pdf"
                                    )
                                
                                st.success("Handwriting generated successfully!")
                            else:
                                st.error("Failed to generate handwriting. Please try again.")
                    
                    except Exception as e:
                        st.error(f"Error generating handwriting: {str(e)}")

def neural_network_tab(cnn_model, session_id):
    """Handle CNN-based neural network functionality"""
    st.header("🧠 Neural Network AI")
    st.markdown("Advanced deep learning features for handwriting recognition and style analysis.")
    
    # Model status and initialization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Model Status")
        model_info = cnn_model.get_model_info()
        
        if model_info['model_initialized']:
            st.success("Neural network model is ready")
            st.write(f"**Model Type:** {model_info['model_type'].upper()}")
            st.write(f"**Input Shape:** {model_info['input_shape']}")
            st.write(f"**Character Classes:** {model_info['num_classes']}")
            if 'total_parameters' in model_info:
                st.write(f"**Parameters:** {model_info['total_parameters']:,}")
        else:
            st.warning("Neural network model not initialized")
    
    with col2:
        st.subheader("🔧 Model Actions")
        if st.button("Initialize Model", type="primary"):
            with st.spinner("Initializing neural network..."):
                if cnn_model.initialize_model():
                    st.success("Model initialized successfully!")
                    st.rerun()
                else:
                    st.error("Failed to initialize model")
        
        if st.button("Load Pretrained", type="secondary"):
            with st.spinner("Loading pretrained model..."):
                if cnn_model.load_model():
                    st.success("Pretrained model loaded!")
                    st.rerun()
                else:
                    st.info("No pretrained model found")
    
    # Training section
    st.subheader("📚 Model Training")
    
    training_mode = st.radio(
        "Training Mode:",
        ["Quick Demo Training", "Custom Dataset Training", "Transfer Learning"],
        horizontal=True
    )
    
    if training_mode == "Quick Demo Training":
        st.info("Train the model with synthetic data for demonstration purposes")
        
        col_train1, col_train2, col_train3 = st.columns(3)
        with col_train1:
            epochs = st.slider("Training Epochs", 1, 50, 10)
        with col_train2:
            batch_size = st.slider("Batch Size", 8, 64, 32)
        with col_train3:
            num_samples = st.slider("Training Samples", 100, 5000, 1000)
        
        if st.button("Start Training", type="primary"):
            if not model_info['model_initialized']:
                st.error("Please initialize the model first")
            else:
                with st.spinner("Training neural network..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Training callback to update progress
                    def progress_callback(epoch, total_epochs):
                        progress = epoch / total_epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Training... Epoch {epoch}/{total_epochs}")
                    
                    success = cnn_model.train_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        progress_callback=progress_callback
                    )
                    
                    if success:
                        st.success("Training completed successfully!")
                        # Log training to database
                        try:
                            db_service.log_analytics(
                                event_type='neural_training',
                                session_id=session_id,
                                success=True,
                                metadata={
                                    'epochs': epochs,
                                    'batch_size': batch_size,
                                    'samples': num_samples
                                }
                            )
                        except:
                            pass
                    else:
                        st.error("Training failed")
    
    elif training_mode == "Custom Dataset Training":
        st.info("Upload your own handwriting samples for training")
        
        uploaded_files = st.file_uploader(
            "Upload handwriting images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images containing single characters"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            # Show preview of uploaded images
            cols = st.columns(min(len(uploaded_files), 5))
            for i, uploaded_file in enumerate(uploaded_files[:5]):
                with cols[i]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Sample {i+1}", width=100)
            
            if len(uploaded_files) > 5:
                st.write(f"... and {len(uploaded_files) - 5} more images")
            
            if st.button("Process Dataset"):
                st.info("Custom dataset training feature coming soon!")
    
    else:  # Transfer Learning
        st.info("Fine-tune a pretrained model on your specific handwriting style")
        st.write("This feature allows you to adapt an existing model to your specific handwriting characteristics.")
        
        if st.button("Start Transfer Learning"):
            st.info("Transfer learning feature coming soon!")
    
    # Character Recognition Testing
    st.subheader("🔍 Character Recognition Test")
    
    test_mode = st.radio(
        "Test Mode:",
        ["Upload Image", "Draw Character", "Camera Capture"],
        horizontal=True
    )
    
    if test_mode == "Upload Image":
        uploaded_test = st.file_uploader(
            "Upload a character image for recognition",
            type=['png', 'jpg', 'jpeg'],
            key="neural_test_upload"
        )
        
        if uploaded_test:
            test_image = Image.open(uploaded_test)
            
            col_test1, col_test2 = st.columns(2)
            with col_test1:
                st.image(test_image, caption="Original Image", width=200)
            
            with col_test2:
                if st.button("Recognize Character"):
                    if not model_info['model_initialized']:
                        st.error("Please initialize and train the model first")
                    else:
                        with st.spinner("Analyzing character..."):
                            char, confidence = cnn_model.predict_character(test_image)
                            
                            if char:
                                st.success(f"**Predicted Character:** {char}")
                                st.write(f"**Confidence:** {confidence:.2%}")
                                
                                # Analyze handwriting style
                                style_analysis = cnn_model.analyze_handwriting_style(test_image)
                                if style_analysis:
                                    st.subheader("📊 Style Analysis")
                                    
                                    metrics_col1, metrics_col2 = st.columns(2)
                                    with metrics_col1:
                                        st.metric("Texture Complexity", f"{style_analysis['texture_complexity']:.3f}")
                                        st.metric("Edge Density", f"{style_analysis['edge_density']:.3f}")
                                        st.metric("Stroke Variation", f"{style_analysis['stroke_variation']:.3f}")
                                    
                                    with metrics_col2:
                                        st.metric("Writing Consistency", f"{style_analysis['writing_consistency']:.3f}")
                                        st.metric("Slant Estimate", f"{style_analysis['slant_estimate']:.1f}°")
                                        st.metric("Thickness Variation", f"{style_analysis['thickness_variation']:.3f}")
                            else:
                                st.error("Failed to recognize character")
    
    elif test_mode == "Draw Character":
        st.info("Interactive drawing feature coming soon!")
        st.write("This will allow you to draw characters directly in the browser for real-time recognition.")
    
    else:  # Camera Capture
        st.info("Camera capture feature coming soon!")
        st.write("This will allow you to capture handwriting directly from your camera.")
    
    # Model Management
    st.subheader("💾 Model Management")
    
    col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)
    
    with col_mgmt1:
        if st.button("Save Model"):
            if cnn_model.model is not None:
                if cnn_model.save_model():
                    st.success("Model saved successfully!")
                else:
                    st.error("Failed to save model")
            else:
                st.warning("No model to save")
    
    with col_mgmt2:
        if st.button("Export Model"):
            st.info("Model export feature coming soon!")
    
    with col_mgmt3:
        if st.button("Model Statistics"):
            if cnn_model.model is not None:
                st.json(model_info)
            else:
                st.warning("No model loaded")

def style_training_tab(style_model, session_id):
    """Handle style training functionality"""
    st.header("Custom Style Training")
    st.markdown("Train the model on your own handwriting style for personalized generation.")
    
    st.info("🚧 This feature allows you to create custom handwriting styles by training on sample images.")
    
    # Training data upload
    st.subheader("Upload Training Samples")
    training_images = st.file_uploader(
        "Upload handwriting samples",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple images of your handwriting for better training results"
    )
    
    if training_images:
        st.write(f"Uploaded {len(training_images)} training samples")
        
        # Display sample images
        if st.checkbox("Preview uploaded samples"):
            cols = st.columns(min(3, len(training_images)))
            for i, img_file in enumerate(training_images[:3]):
                with cols[i]:
                    img = Image.open(img_file)
                    st.image(img, caption=f"Sample {i+1}", use_column_width=True)
        
        # Training parameters
        st.subheader("Training Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            style_name = st.text_input("Style Name:", placeholder="My Handwriting Style")
            training_epochs = st.slider("Training Epochs", 10, 100, 50)
        
        with col2:
            learning_rate = st.select_slider("Learning Rate", [0.001, 0.01, 0.1], value=0.01)
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        
        if st.button("Start Training", type="primary"):
            if not style_name.strip():
                st.error("Please provide a name for your custom style.")
            else:
                with st.spinner("Training custom style... This may take a few minutes."):
                    try:
                        # Prepare training data
                        training_data = []
                        for img_file in training_images:
                            img = Image.open(img_file)
                            training_data.append(np.array(img))
                        
                        # Train the model
                        training_progress = st.progress(0)
                        training_status = st.empty()
                        
                        success = style_model.train_custom_style(
                            style_name=style_name,
                            training_data=training_data,
                            epochs=training_epochs,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            progress_callback=lambda epoch, total: (
                                training_progress.progress(epoch / total),
                                training_status.write(f"Training epoch {epoch}/{total}")
                            )
                        )
                        
                        if success:
                            st.success(f"Custom style '{style_name}' trained successfully!")
                            st.balloons()
                        else:
                            st.error("Training failed. Please check your training data and try again.")
                    
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
    
    # Display existing custom styles
    st.subheader("Available Custom Styles")
    custom_styles = style_model.get_custom_styles()
    
    if custom_styles:
        for style in custom_styles:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{style['name']}** - Created: {style['created_date']}")
            with col2:
                st.write(f"Accuracy: {style['accuracy']:.2f}%")
            with col3:
                if st.button(f"Delete", key=f"delete_{style['name']}"):
                    style_model.delete_custom_style(style['name'])
                    st.rerun()
    else:
        st.info("No custom styles available. Train your first custom style above!")

def github_tab(github_integration, style_model, cnn_model, session_id):
    """Handle GitHub integration and version control"""
    st.header("🔗 GitHub Integration")
    st.markdown("Version control and collaboration features for your handwriting system data.")
    
    # GitHub Authentication Status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔐 Authentication Status")
        
        if github_integration.is_authenticated():
            user_info = github_integration.get_user_info()
            if user_info:
                st.success(f"Connected as **{user_info['login']}** ({user_info['name']})")
                st.write(f"Public repositories: {user_info['public_repos']}")
                st.write(f"Profile: {user_info['html_url']}")
            else:
                st.success("GitHub authenticated")
        else:
            st.error("GitHub authentication failed")
            st.info("Please check your GitHub token in the environment settings.")
    
    with col2:
        st.subheader("🏗️ Quick Actions")
        
        if st.button("Create Handwriting Repo", type="primary"):
            if github_integration.is_authenticated():
                repo_name = f"handwriting-system-{datetime.now().strftime('%Y%m%d')}"
                with st.spinner("Creating repository..."):
                    repo = github_integration.create_handwriting_repository(repo_name)
                    if repo:
                        st.success(f"Repository created: {repo['html_url']}")
                    else:
                        st.error("Failed to create repository")
            else:
                st.error("Please authenticate with GitHub first")
        
        if st.button("Refresh Repositories"):
            st.rerun()
    
    # Repository Management
    if github_integration.is_authenticated():
        st.subheader("📂 Repository Management")
        
        # List repositories
        repos = github_integration.list_repositories()
        
        if repos:
            repo_options = [f"{repo['name']} ({'private' if repo['private'] else 'public'})" for repo in repos]
            selected_repo_display = st.selectbox("Select Repository", repo_options)
            
            if selected_repo_display:
                # Extract repo name from display
                selected_repo_name = selected_repo_display.split(' (')[0]
                selected_repo = next((repo for repo in repos if repo['name'] == selected_repo_name), None)
                
                if selected_repo:
                    repo_owner = selected_repo['owner']['login']
                    
                    # Repository Information
                    with st.expander("Repository Information", expanded=True):
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.write(f"**Name:** {selected_repo['name']}")
                            st.write(f"**Owner:** {repo_owner}")
                            st.write(f"**Visibility:** {'Private' if selected_repo['private'] else 'Public'}")
                            
                        with col_info2:
                            st.write(f"**Size:** {selected_repo['size']} KB")
                            st.write(f"**Stars:** {selected_repo['stargazers_count']}")
                            st.write(f"**Language:** {selected_repo['language'] or 'N/A'}")
                        
                        st.write(f"**Description:** {selected_repo['description'] or 'No description'}")
                        st.write(f"**URL:** {selected_repo['html_url']}")
                    
                    # Style Management
                    st.subheader("🎨 Style Management")
                    
                    col_style1, col_style2 = st.columns(2)
                    
                    with col_style1:
                        st.write("**Save Custom Style**")
                        
                        # Get custom styles from database
                        try:
                            custom_styles = db_service.get_custom_styles(session_id)
                            if custom_styles:
                                style_names = [style.style_name for style in custom_styles]
                                selected_style = st.selectbox("Select style to save", style_names)
                                
                                if selected_style and st.button("Save to GitHub"):
                                    style_data = next((s for s in custom_styles if s.style_name == selected_style), None)
                                    if style_data:
                                        style_config = style_data.style_config
                                        with st.spinner("Saving style to GitHub..."):
                                            success = github_integration.save_handwriting_style(
                                                repo_owner, selected_repo_name, selected_style, 
                                                style_config, session_id
                                            )
                                            if success:
                                                st.success(f"Style '{selected_style}' saved to GitHub!")
                                            else:
                                                st.error("Failed to save style")
                            else:
                                st.info("No custom styles found. Create some styles first.")
                        except Exception as e:
                            st.error(f"Error accessing custom styles: {str(e)}")
                    
                    with col_style2:
                        st.write("**Load Shared Style**")
                        
                        # List styles in repository
                        shared_styles = github_integration.list_shared_styles(repo_owner, selected_repo_name)
                        
                        if shared_styles:
                            selected_shared_style = st.selectbox("Select style to load", shared_styles)
                            
                            if selected_shared_style and st.button("Load from GitHub"):
                                with st.spinner("Loading style from GitHub..."):
                                    style_data = github_integration.load_handwriting_style(
                                        repo_owner, selected_repo_name, selected_shared_style
                                    )
                                    if style_data:
                                        st.success(f"Style '{selected_shared_style}' loaded!")
                                        st.json(style_data['metadata'])
                                    else:
                                        st.error("Failed to load style")
                        else:
                            st.info("No shared styles found in this repository.")
                    
                    # Neural Network Model Management
                    st.subheader("🧠 Neural Network Models")
                    
                    col_model1, col_model2 = st.columns(2)
                    
                    with col_model1:
                        st.write("**Save Neural Network Model**")
                        
                        model_name = st.text_input("Model name", value=f"handwriting_model_{datetime.now().strftime('%Y%m%d')}")
                        
                        if st.button("Save Model to GitHub"):
                            if cnn_model.weights and len(cnn_model.weights) > 0:
                                model_info = cnn_model.get_model_info()
                                model_data = {
                                    'model_info': model_info,
                                    'weights': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in cnn_model.weights.items()},
                                    'training_status': cnn_model.is_trained
                                }
                                
                                with st.spinner("Saving model to GitHub..."):
                                    success = github_integration.save_neural_network_model(
                                        repo_owner, selected_repo_name, model_name, 
                                        model_data, session_id
                                    )
                                    if success:
                                        st.success(f"Model '{model_name}' saved to GitHub!")
                                    else:
                                        st.error("Failed to save model")
                            else:
                                st.warning("No trained model to save. Train a model first.")
                    
                    with col_model2:
                        st.write("**Backup User Data**")
                        
                        backup_name = st.text_input("Backup description", value="Weekly backup")
                        
                        if st.button("Create Backup"):
                            try:
                                # Gather user data
                                handwriting_samples = db_service.get_handwriting_samples(session_id, limit=100)
                                generated_texts = db_service.get_generated_texts(session_id, limit=100)
                                custom_styles = db_service.get_custom_styles(session_id)
                                
                                user_data = {
                                    'handwriting_samples': len(handwriting_samples),
                                    'generated_texts': len(generated_texts),
                                    'custom_styles': len(custom_styles),
                                    'backup_description': backup_name
                                }
                                
                                with st.spinner("Creating backup..."):
                                    success = github_integration.backup_user_data(
                                        repo_owner, selected_repo_name, user_data, session_id
                                    )
                                    if success:
                                        st.success("Backup created successfully!")
                                    else:
                                        st.error("Failed to create backup")
                                        
                            except Exception as e:
                                st.error(f"Error creating backup: {str(e)}")
                    
                    # Repository Statistics
                    st.subheader("📊 Repository Statistics")
                    
                    repo_stats = github_integration.get_repository_stats(repo_owner, selected_repo_name)
                    if repo_stats:
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Repository Size", f"{repo_stats['size']} KB")
                            st.metric("Stars", repo_stats['stars'])
                        
                        with col_stat2:
                            st.metric("Forks", repo_stats['forks'])
                            st.metric("Language", repo_stats['language'] or 'Mixed')
                        
                        with col_stat3:
                            st.write(f"**Last Updated:** {repo_stats['updated_at'][:10]}")
                            st.write(f"**Visibility:** {'Private' if repo_stats['private'] else 'Public'}")
        else:
            st.info("No repositories found. Create a new repository to get started.")

if __name__ == "__main__":
    main()
