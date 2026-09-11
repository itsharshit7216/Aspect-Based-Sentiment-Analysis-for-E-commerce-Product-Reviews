import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from demo import demo, engine

# Initialize FastAPI App
app = FastAPI(
    title="Aspect-Based Sentiment Analysis API",
    description="End-to-End ABSA REST API for Aspect Term Extraction and Sentiment Classification.",
    version="1.0.0"
)

# Enable CORS for local development and web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestText(BaseModel):
    sentence: str = Field(
        ..., 
        json_schema_extra={"example": "The screen is bright but the speakers are disappointing."},
        description="The product review sentence to analyze."
    )

class ResponseSchema(BaseModel):
    sentence: str
    analysis: dict
    extracted_aspects_count: int

@app.get("/api-info", tags=["Health"])
def root_info():
    return {
        "status": "online",
        "message": "Welcome to Aspect-Based Sentiment Analysis (ABSA) API",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "custom_ate_loaded": engine.is_custom_ate_loaded,
        "custom_asc_loaded": engine.is_custom_asc_loaded
    }

@app.post("/analyze", response_model=ResponseSchema, tags=["ABSA"])
def analyze_text(request: RequestText):
    if not request.sentence.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Sentence input cannot be empty."
        )
    
    result = engine.analyze_sentence(request.sentence)
    return result

# Mount Gradio Web UI directly into FastAPI so both Web UI and REST API run together
import gradio as gr
from demo import demo
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

