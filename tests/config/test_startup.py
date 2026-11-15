import time
import pytest
from fastapi.testclient import TestClient


def test_fastapi_app_cold_startup_latency():
    """Verify FastAPI application cold import and startup completes within 10 seconds (AC3)."""
    start_time = time.perf_counter()
    
    # Import app module
    import app as app_module
    
    # Create test client which triggers app lifespan startup
    with TestClient(app_module.app) as client:
        # Check health or basic response
        response = client.get("/docs")
        assert response.status_code in [200, 404]
    
    elapsed = time.perf_counter() - start_time
    assert elapsed <= 10.0, f"Cold startup took {elapsed:.2f}s, exceeding 10s threshold"
