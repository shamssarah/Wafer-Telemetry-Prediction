from pydantic import BaseModel
from typing import List

class ImageRequest (BaseModel):
    wafer_id : int
    img : List[List[float]]

class ImageResponse (BaseModel):

    anomaly : bool # yes or no
    reconstruction_error : float