from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.analyze import router as analyze_router


app = FastAPI()

#  ADD THIS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # IMPORTANT
    allow_headers=["*"],
)

app.include_router(analyze_router)

@app.get("/")
def home():
    return {"message": "API is running 🚀"}