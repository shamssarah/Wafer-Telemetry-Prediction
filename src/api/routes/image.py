# routes/image.py
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from api.models.image import ImageResponse
from api.pipelines.image import image_pipeline

router = APIRouter()

@router.post("/ingest/image", response_model=ImageResponse)
async def ingest_image(file: UploadFile = File(...)):
    contents = await file.read()          # raw bytes
    img = Image.open(io.BytesIO(contents)) # convert to PIL
    result = image_pipeline(img)
    return result