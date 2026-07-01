import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Import the Pydantic schemas and the core Gemini logic from your agent module
from agent import ChatRequest, ChatResponse, generate_agent_response

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Agentic Recommender", docs_url=None, redoc_url=None)

# --- Middleware (Security & Timeouts) ---
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # SHL's evaluator times out each /chat call at 30s -- keeping this
            # below that (28s) means you see a clean 504 locally instead of
            # the evaluator silently killing the connection past your own
            # unbounded wait, so slow traces stay visible during testing.
            return await asyncio.wait_for(call_next(request), timeout=28.0)
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout triggered on {request.url}")
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out during processing."}
            )

app.add_middleware(TimeoutMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

# --- Global Exception Handlers (Leak Prevention) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request payload format."}
    )

# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Readiness endpoint required by the SHL evaluator."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Stateless conversation endpoint.
    Passes the validated message history to the Gemini agent.
    """
    logger.info(f"Received chat request with {len(request.messages)} messages.")
    
    # Convert Pydantic models to dicts before passing to the agent
    message_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Trigger the Gemini + FAISS RAG pipeline
    response = await generate_agent_response(message_history)
    
    return response