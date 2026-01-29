from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.core.cors import DEV_ALLOWED_ORIGINS
from dotenv import load_dotenv
from app.core.lifespan import lifespan

load_dotenv()
app = FastAPI(title="Portfolio Chat Bot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/v1", tags=["Health"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
