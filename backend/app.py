from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routes import predict, compare, profiles
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Forensic Engine: Starting ML System...")
    # The new ModelManager handles loading gracefully via Singleton when predict is called
    yield
    print("Forensic Engine: Shutdown.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} Forensic API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}

# Register Forensic Routers
app.include_router(predict.router, tags=["Forensics"])
app.include_router(compare.router, tags=["Forensics"])
app.include_router(profiles.router, tags=["Profiles & Datasets"])
