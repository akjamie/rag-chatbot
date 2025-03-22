from contextlib import asynccontextmanager
import time
import redis
import os

from fastapi import FastAPI, Request
from langchain.globals import set_debug
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from api.chat_history_routes import router as chat_history_router
from api.chat_routes import router as chat_router
from api.health_routes import router as health_router
from config.common_settings import CommonConfig
from utils.id_util import get_id
from utils.logging_util import logger, set_context, clear_context
from utils.audit_logger import AuditLogger
from config.database.database_manager import DatabaseManager
from utils.redis_util import RedisUtil

# Global config instance
base_config = CommonConfig()


class LoggingContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get user_id from headers (required)
        user_id = request.headers.get('X-User-Id', 'unknown')
        
        # Get optional session_id and request_id from headers
        session_id = request.headers.get('X-Session-Id')
        request_id = request.headers.get('X-Request-Id')
        
        # Check if this is a chat completion request
        is_chat_completion = request.url.path.endswith('chat/completion')
        
        # Generate session_id if missing or empty for chat completion requests
        if (not session_id or session_id.strip() == "") and is_chat_completion:
            session_id = f"sess_{get_id().lower()}"
            # Remove existing empty header if present
            request.headers._list = [(k, v) for k, v in request.headers._list if k != b'x-session-id']
            # Add new header
            request.headers._list.append(
                (b'x-session-id', session_id.encode())
            )
            logger.debug(f"Generated new session_id: {session_id}")
        
        # Generate request_id if missing or empty for chat completion requests
        if (not request_id or request_id.strip() == "") and is_chat_completion:
            request_id = f"req_{get_id().lower()}"
            # Remove existing empty header if present
            request.headers._list = [(k, v) for k, v in request.headers._list if k != b'x-request-id']
            # Add new header
            request.headers._list.append(
                (b'x-request-id', request_id.encode())
            )
            logger.debug(f"Generated new request_id: {request_id}")
        
        # Update the headers scope for ASGI
        request.scope['headers'] = request.headers._list

        # Set context using keyword arguments
        set_context(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            request_path=request.url.path,
            request_method=request.method,
            start_time=start_time
        )

        try:
            response = await call_next(request)
            # Add IDs to response headers only if they exist
            if session_id:
                response.headers['X-Session-Id'] = session_id
            if request_id:
                response.headers['X-Request-Id'] = request_id
            if user_id:
                response.headers['X-User-Id'] = user_id
            # Add timing information
            request_time = time.time() - start_time
            set_context(request_time_ms=int(request_time * 1000))
            logger.info(f"Request completed in {request_time:.2f}s")
            return response
        finally:
            clear_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application"""
    embedding_job = None
    try:
        # Startup
        logger.info("Initializing application...")

        # Initialize config and setup proxy first
        proxy_result = await base_config.asetup_proxy()
        logger.info(f"Proxy setup {'enabled' if proxy_result else 'disabled'}")

        # Initialize other components (make this non-blocking)

        logger.info("Application startup completed")

        # Important: yield here to let FastAPI take control
        yield

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        # Shutdown
        if embedding_job and embedding_job.scheduler:
            embedding_job.scheduler.shutdown()
        logger.info("Shutting down application...")


app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG Chatbot",
    version="0.1.0",
    lifespan=lifespan
)
app.add_middleware(LoggingContextMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],  # Allow all headers, adjust as needed
    expose_headers=["X-Session-Id", "X-Request-Id", "X-User-Id"],
)

app.include_router(chat_history_router, prefix="/chat")
app.include_router(chat_router, prefix="/chat")
app.include_router(health_router, prefix="/health")

# 应用启动前初始化
@app.on_event("startup")
async def startup_event():
    """Application startup initialization"""
    try:
        # Initialize database tables
        config = CommonConfig()
        AuditLogger.init_database(config)
        
        # Initialize Redis and import Q&A data
        redis_client = redis.Redis(
            host=config.get_env_variable("REDIS_HOST", "localhost"),
            port=int(config.get_env_variable("REDIS_PORT", 6379)),
            db=int(config.get_env_variable("REDIS_DB", 0)),
            decode_responses=True
        )
        
        # Test Redis connection
        redis_client.ping()
        logger.info("Successfully connected to Redis")
        
        # Initialize RedisUtil and import Q&A data
        redis_util = RedisUtil(redis_client)
        qa_file_path = os.path.join(BASE_DIR, "data", "qa_pairs.json")
        imported_count = redis_util.import_from_json(qa_file_path)
        logger.info(f"Successfully imported {imported_count} Q&A pairs from {qa_file_path}")
        
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    # set_debug(True)
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
