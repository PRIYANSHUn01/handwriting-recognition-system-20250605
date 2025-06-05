from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, LargeBinary, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    """User sessions and preferences"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON, default={})

class HandwritingSample(Base):
    """Stored handwriting samples for analysis"""
    __tablename__ = 'handwriting_samples'
    
    id = Column(Integer, primary_key=True)
    user_session = Column(String(100), nullable=False)
    image_data = Column(LargeBinary, nullable=False)
    image_format = Column(String(10), default='PNG')
    extracted_text = Column(Text)
    confidence_score = Column(Float)
    language = Column(String(10), default='eng')
    processing_options = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class GeneratedText(Base):
    """Generated handwriting samples"""
    __tablename__ = 'generated_text'
    
    id = Column(Integer, primary_key=True)
    user_session = Column(String(100), nullable=False)
    input_text = Column(Text, nullable=False)
    style_name = Column(String(50), nullable=False)
    generation_params = Column(JSON)
    image_data = Column(LargeBinary, nullable=False)
    image_format = Column(String(10), default='PNG')
    created_at = Column(DateTime, default=datetime.utcnow)

class CustomStyle(Base):
    """Custom trained handwriting styles"""
    __tablename__ = 'custom_styles'
    
    id = Column(Integer, primary_key=True)
    user_session = Column(String(100), nullable=False)
    style_name = Column(String(100), nullable=False)
    style_config = Column(JSON, nullable=False)
    training_samples_count = Column(Integer, default=0)
    accuracy_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemAnalytics(Base):
    """System usage analytics"""
    __tablename__ = 'system_analytics'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)  # 'recognition', 'generation', 'training'
    user_session = Column(String(100), nullable=False)
    processing_time = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    event_metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ProcessingQueue(Base):
    """Queue for batch processing tasks"""
    __tablename__ = 'processing_queue'
    
    id = Column(Integer, primary_key=True)
    user_session = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)  # 'batch_generation', 'style_training'
    task_data = Column(JSON, nullable=False)
    status = Column(String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    progress = Column(Float, default=0.0)
    result_data = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

# Database connection and session management
def get_database_engine():
    """Get database engine from environment variables"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(database_url)
    return engine

def create_tables():
    """Create all database tables"""
    engine = get_database_engine()
    Base.metadata.create_all(engine)
    return engine

def get_database_session():
    """Get database session"""
    engine = get_database_engine()
    Session = sessionmaker(bind=engine)
    return Session()