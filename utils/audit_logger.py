import json
import time
from typing import Optional, Dict, Any, List, Callable
from sqlalchemy import Column, String, DateTime, Integer, Text, inspect
from sqlalchemy.ext.declarative import declarative_base
import datetime
from config.database.database_manager import DatabaseManager
from utils.logging_util import logger
from fastapi import BackgroundTasks

Base = declarative_base()

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), nullable=False, index=True)
    step = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # START, END, ERROR
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    details = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<AuditLog(request_id='{self.request_id}', step='{self.step}', status='{self.status}')>"


class AuditLogger:
    """Audit Logger for tracking workflow steps and performance with FastAPI BackgroundTasks"""
    
    # Global list to store pending logs when no BackgroundTasks is available
    _pending_logs: List[Callable] = []
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._ensure_table_exists()
        logger.info("Audit logger initialized with FastAPI BackgroundTasks support")
    
    @staticmethod
    def initialize_tables(engine):
        """Initialize audit log tables"""
        inspector = inspect(engine)
        if not inspector.has_table(AuditLog.__tablename__):
            logger.info(f"Creating audit log table: {AuditLog.__tablename__}")
            Base.metadata.create_all(engine, tables=[AuditLog.__table__])
            logger.info("Audit log table created successfully")
        else:
            logger.info(f"Audit log table already exists: {AuditLog.__tablename__}")
    
    def _ensure_table_exists(self):
        """Ensure audit log table exists"""
        try:
            AuditLogger.initialize_tables(self.db_manager.engine)
        except Exception as e:
            logger.error(f"Failed to initialize audit log table: {e}")
    
    @classmethod
    def init_database(cls, config):
        """Initialize database tables - called on application startup"""
        try:
            # Get the database URI instead of the manager
            postgres_uri = config.get_db_manager()
            
            # Create a database manager with the URI
            db_manager = DatabaseManager(postgres_uri)
            
            # Initialize audit log tables
            cls.initialize_tables(db_manager.engine)
            logger.info("Audit log database tables initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audit log database: {e}")
            return False
    
    def _write_log_entry(self, log_entry):
        """Write a log entry directly to the database (synchronous)"""
        try:
            with self.db_manager.session() as session:
                session.add(log_entry)
                session.commit()
            logger.debug(f"Audit log written: {log_entry.step} - {log_entry.status}")
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    @classmethod
    def add_pending_log(cls, log_func: Callable):
        """Add a pending log to the global list"""
        cls._pending_logs.append(log_func)
        logger.debug(f"Added pending log, total pending: {len(cls._pending_logs)}")
    
    @classmethod
    def process_pending_logs(cls, background_tasks: BackgroundTasks):
        """Process all pending logs"""
        if not cls._pending_logs:
            return
        
        count = len(cls._pending_logs)
        for log_func in cls._pending_logs:
            background_tasks.add_task(log_func)
        
        cls._pending_logs.clear()
        logger.debug(f"Scheduled {count} pending logs for processing")
    
    def log_step(self, request_id: str, user_id: str, session_id: str, 
                step: str, status: str, details: Optional[Dict[str, Any]] = None, 
                background_tasks: Optional[BackgroundTasks] = None):
        """Log a step - using background_tasks if available"""
        try:
            # Create log entry
            log_entry = AuditLog(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                step=step,
                status=status,
                details=json.dumps(details) if details else None
            )
            
            if background_tasks is not None:
                # Add to background tasks for non-blocking execution
                background_tasks.add_task(self._write_log_entry, log_entry)
                logger.debug(f"Audit log added to background tasks: {step} - {status}")
            else:
                # No background_tasks available, add to pending logs
                self.add_pending_log(lambda: self._write_log_entry(log_entry))
                logger.debug(f"Audit log added to pending logs: {step} - {status}")
            
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
    
    def start_step(self, request_id: str, user_id: str, session_id: str, 
                  step: str, details: Optional[Dict[str, Any]] = None,
                  background_tasks: Optional[BackgroundTasks] = None):
        """Log step start - non-blocking if background_tasks is provided"""
        logger.info(f"Recording audit log start step: {step}")
        self.log_step(request_id, user_id, session_id, step, "START", details, background_tasks)
        return time.time()  # Return start time for calculating execution time
    
    def end_step(self, request_id: str, user_id: str, session_id: str, 
                step: str, start_time: float, details: Optional[Dict[str, Any]] = None,
                background_tasks: Optional[BackgroundTasks] = None):
        """Log step end - non-blocking if background_tasks is provided"""
        execution_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
        if details is None:
            details = {}
        
        # Put execution time at the beginning of details
        execution_details = {
            "execution_time_ms": execution_time
        }
        execution_details.update(details)  # Add other details
        
        self.log_step(request_id, user_id, session_id, step, "END", execution_details, background_tasks)
    
    def error_step(self, request_id: str, user_id: str, session_id: str, 
                  step: str, error: Exception, details: Optional[Dict[str, Any]] = None,
                  background_tasks: Optional[BackgroundTasks] = None):
        """Log step error - non-blocking if background_tasks is provided"""
        if details is None:
            details = {}
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        error_details.update(details)
        
        self.log_step(request_id, user_id, session_id, step, "ERROR", error_details, background_tasks)
    
    def shutdown(self):
        """Shutdown the audit logger"""
        logger.info("Audit logger shutdown complete")


# This middleware function can be added to FastAPI application to handle pending logs
def process_pending_audit_logs(request, call_next):
    """Middleware to process any pending audit logs using request's background tasks"""
    response = call_next(request)
    
    # Get the request's background_tasks
    if hasattr(request.state, "background_tasks"):
        background_tasks = request.state.background_tasks
        AuditLogger.process_pending_logs(background_tasks)
        
    return response


# Singleton pattern for global audit logger
_audit_logger = None

def get_audit_logger(db_manager: DatabaseManager = None) -> AuditLogger:
    """Get audit logger instance"""
    global _audit_logger
    if _audit_logger is None and db_manager is not None:
        _audit_logger = AuditLogger(db_manager)
    return _audit_logger 