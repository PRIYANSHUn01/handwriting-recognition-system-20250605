import io
import time
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.exc import SQLAlchemyError
from database.models import (
    User, HandwritingSample, GeneratedText, CustomStyle, 
    SystemAnalytics, ProcessingQueue, get_database_session, create_tables
)

class DatabaseService:
    """Service layer for database operations"""
    
    def __init__(self):
        self.session = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            create_tables()
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
    
    def get_session(self):
        """Get database session"""
        if not self.session:
            self.session = get_database_session()
        return self.session
    
    def close_session(self):
        """Close database session"""
        if self.session:
            self.session.close()
            self.session = None
    
    # User Management
    def create_user_session(self, session_id: str) -> User:
        """Create or update user session"""
        try:
            session = self.get_session()
            user = session.query(User).filter_by(session_id=session_id).first()
            
            if user:
                user.last_active = datetime.utcnow()
            else:
                user = User(session_id=session_id)
                session.add(user)
            
            session.commit()
            return user
        except SQLAlchemyError as e:
            print(f"Error creating user session: {str(e)}")
            return None
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            session = self.get_session()
            user = session.query(User).filter_by(session_id=session_id).first()
            
            if user:
                user.preferences = preferences
                user.last_active = datetime.utcnow()
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            print(f"Error updating preferences: {str(e)}")
            return False
    
    # Handwriting Sample Management
    def save_handwriting_sample(self, session_id: str, image_data: bytes, 
                               extracted_text: str, confidence: float, 
                               language: str, processing_options: Dict[str, Any]) -> int:
        """Save handwriting recognition sample"""
        try:
            session = self.get_session()
            sample = HandwritingSample(
                user_session=session_id,
                image_data=image_data,
                extracted_text=extracted_text,
                confidence_score=confidence,
                language=language,
                processing_options=processing_options
            )
            session.add(sample)
            session.commit()
            return sample.id
        except SQLAlchemyError as e:
            print(f"Error saving handwriting sample: {str(e)}")
            return None
    
    def get_handwriting_samples(self, session_id: str, limit: int = 10) -> List[HandwritingSample]:
        """Get user's handwriting samples"""
        try:
            session = self.get_session()
            samples = session.query(HandwritingSample)\
                           .filter_by(user_session=session_id)\
                           .order_by(HandwritingSample.created_at.desc())\
                           .limit(limit).all()
            return samples
        except SQLAlchemyError as e:
            print(f"Error getting handwriting samples: {str(e)}")
            return []
    
    # Generated Text Management
    def save_generated_text(self, session_id: str, input_text: str, style_name: str,
                           generation_params: Dict[str, Any], image_data: bytes) -> int:
        """Save generated handwriting"""
        try:
            session = self.get_session()
            generated = GeneratedText(
                user_session=session_id,
                input_text=input_text,
                style_name=style_name,
                generation_params=generation_params,
                image_data=image_data
            )
            session.add(generated)
            session.commit()
            return generated.id
        except SQLAlchemyError as e:
            print(f"Error saving generated text: {str(e)}")
            return None
    
    def get_generated_texts(self, session_id: str, limit: int = 10) -> List[GeneratedText]:
        """Get user's generated texts"""
        try:
            session = self.get_session()
            texts = session.query(GeneratedText)\
                         .filter_by(user_session=session_id)\
                         .order_by(GeneratedText.created_at.desc())\
                         .limit(limit).all()
            return texts
        except SQLAlchemyError as e:
            print(f"Error getting generated texts: {str(e)}")
            return []
    
    # Custom Style Management
    def save_custom_style(self, session_id: str, style_name: str, style_config: Dict[str, Any],
                         samples_count: int, accuracy: float) -> int:
        """Save custom trained style"""
        try:
            session = self.get_session()
            
            # Check if style exists for this user
            existing_style = session.query(CustomStyle)\
                                  .filter_by(user_session=session_id, style_name=style_name)\
                                  .first()
            
            if existing_style:
                existing_style.style_config = style_config
                existing_style.training_samples_count = samples_count
                existing_style.accuracy_score = accuracy
                existing_style.updated_at = datetime.utcnow()
                style_id = existing_style.id
            else:
                style = CustomStyle(
                    user_session=session_id,
                    style_name=style_name,
                    style_config=style_config,
                    training_samples_count=samples_count,
                    accuracy_score=accuracy
                )
                session.add(style)
                session.flush()
                style_id = style.id
            
            session.commit()
            return style_id
        except SQLAlchemyError as e:
            print(f"Error saving custom style: {str(e)}")
            return None
    
    def get_custom_styles(self, session_id: str) -> List[CustomStyle]:
        """Get user's custom styles"""
        try:
            session = self.get_session()
            styles = session.query(CustomStyle)\
                          .filter_by(user_session=session_id, is_active=True)\
                          .order_by(CustomStyle.created_at.desc()).all()
            return styles
        except SQLAlchemyError as e:
            print(f"Error getting custom styles: {str(e)}")
            return []
    
    def delete_custom_style(self, session_id: str, style_name: str) -> bool:
        """Delete custom style"""
        try:
            session = self.get_session()
            style = session.query(CustomStyle)\
                         .filter_by(user_session=session_id, style_name=style_name)\
                         .first()
            
            if style:
                style.is_active = False
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            print(f"Error deleting custom style: {str(e)}")
            return False
    
    # Analytics
    def log_analytics(self, event_type: str, session_id: str, processing_time: float = None,
                     success: bool = True, error_message: str = None, 
                     metadata: Dict[str, Any] = None) -> int:
        """Log system analytics"""
        try:
            session = self.get_session()
            analytics = SystemAnalytics(
                event_type=event_type,
                user_session=session_id,
                processing_time=processing_time,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            session.add(analytics)
            session.commit()
            return analytics.id
        except SQLAlchemyError as e:
            print(f"Error logging analytics: {str(e)}")
            return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            session = self.get_session()
            
            # Total operations
            total_recognitions = session.query(SystemAnalytics)\
                                      .filter_by(event_type='recognition').count()
            total_generations = session.query(SystemAnalytics)\
                                     .filter_by(event_type='generation').count()
            total_trainings = session.query(SystemAnalytics)\
                                   .filter_by(event_type='training').count()
            
            # Success rates
            successful_recognitions = session.query(SystemAnalytics)\
                                           .filter_by(event_type='recognition', success=True).count()
            successful_generations = session.query(SystemAnalytics)\
                                          .filter_by(event_type='generation', success=True).count()
            
            # Active users (last 24 hours)
            from datetime import timedelta
            yesterday = datetime.utcnow() - timedelta(days=1)
            active_users = session.query(User)\
                                .filter(User.last_active >= yesterday).count()
            
            return {
                'total_recognitions': total_recognitions,
                'total_generations': total_generations,
                'total_trainings': total_trainings,
                'recognition_success_rate': successful_recognitions / max(total_recognitions, 1) * 100,
                'generation_success_rate': successful_generations / max(total_generations, 1) * 100,
                'active_users_24h': active_users
            }
        except SQLAlchemyError as e:
            print(f"Error getting system stats: {str(e)}")
            return {}
    
    # Processing Queue
    def add_processing_task(self, session_id: str, task_type: str, task_data: Dict[str, Any]) -> int:
        """Add task to processing queue"""
        try:
            session = self.get_session()
            task = ProcessingQueue(
                user_session=session_id,
                task_type=task_type,
                task_data=task_data
            )
            session.add(task)
            session.commit()
            return task.id
        except SQLAlchemyError as e:
            print(f"Error adding processing task: {str(e)}")
            return None
    
    def update_task_progress(self, task_id: int, progress: float, status: str = None) -> bool:
        """Update task progress"""
        try:
            session = self.get_session()
            task = session.query(ProcessingQueue).filter_by(id=task_id).first()
            
            if task:
                task.progress = progress
                if status:
                    task.status = status
                if status == 'processing' and not task.started_at:
                    task.started_at = datetime.utcnow()
                elif status in ['completed', 'failed']:
                    task.completed_at = datetime.utcnow()
                
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            print(f"Error updating task progress: {str(e)}")
            return False

# Global database service instance
db_service = DatabaseService()