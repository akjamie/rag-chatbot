import traceback
from typing import Any, Dict

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.common_settings import CommonConfig
from handler.generic_query_handler import QueryHandler
from utils.id_util import get_id
from utils.logging_util import logger

router = APIRouter(tags=['chat'])
base_config = CommonConfig()


class QueryRequest(BaseModel):
    user_input: str


class QueryResponse(BaseModel):
    data: Dict[str, Any]
    user_input: str


@router.post("/completion", response_model=QueryResponse)
def process_query(
        request: QueryRequest,
        x_user_id: str = Header(...),
        x_session_id: str = Header(default=None),
        x_request_id: str = Header(default=None),
        authorization: str | None = Header(default=None)
):
    logger.info(f"Received query: {request.user_input}")

    try:
        query_handler = QueryHandler(
            llm=base_config.get_model("chatllm"),
            vector_store=base_config.get_vector_store(),
            config=base_config
        )
        result = query_handler.handle(
            user_input=request.user_input,
            user_id=x_user_id,
            session_id=x_session_id,
            request_id=x_request_id
        )

        return JSONResponse(content=QueryResponse(
            data=result,
            user_input=request.user_input
        ).dict(), headers={"X-User-Id": x_user_id, "X-Session-Id": x_session_id, "X-Request-Id": x_request_id})

    except ValueError as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "error_message": str(e),
                "error_code": "BAD_REQUEST"
            },
            status_code=400, headers={"X-User-Id": x_user_id, "X-Session-Id": x_session_id, "X-Request-Id": x_request_id}
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "error_message": str(e),
                "error_code": "INTERNAL_SERVER_ERROR"
            },
            status_code=500,
            headers={"X-User-Id": x_user_id, "X-Session-Id": x_session_id, "X-Request-Id": x_request_id}
        )


@router.post("/stream")
async def stream_query(
        request: QueryRequest,
        x_user_id: str = Header(...),
        x_session_id: str = Header(default=None),
        x_request_id: str = Header(default=None),
        authorization: str | None = Header(default=None)
):
    """Stream chat completion responses"""
    logger.info(f"Received streaming query: {request.user_input}")

    try:
        query_handler = QueryHandler(
            llm=base_config.get_model("chatllm"),
            vector_store=base_config.get_vector_store(),
            config=base_config
        )

        return StreamingResponse(
            query_handler.handle_stream(
                user_input=request.user_input,
                user_id=x_user_id,
                session_id=x_session_id,
                request_id=x_request_id
            ),
            media_type='text/event-stream',
            headers={
                "X-User-Id": x_user_id,
                "X-Session-Id": x_session_id,
                "X-Request-Id": x_request_id
            }
        )

    except ValueError as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "error_message": str(e),
                "error_code": "BAD_REQUEST"
            },
            status_code=400,
            headers={"X-User-Id": x_user_id, "X-Session-Id": x_session_id, "X-Request-Id": x_request_id}
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "error_message": str(e),
                "error_code": "INTERNAL_SERVER_ERROR"
            },
            status_code=500,
            headers={"X-User-Id": x_user_id, "X-Session-Id": x_session_id, "X-Request-Id": x_request_id}
        )


if __name__ == "__main__":
    base_config.setup_proxy()

    from langchain.globals import set_debug

    set_debug(True)
    query_handler = QueryHandler(
        llm=base_config.get_model("chatllm"),
        vector_store=base_config.get_vector_store(),
        config=base_config
    )
    response = query_handler.handle(user_input="What is the capital of France?", user_id="test", session_id="test",
                                    request_id=get_id())
    print(response)
