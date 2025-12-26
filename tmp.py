from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

@app.post("/image")
async def handle_image(file: UploadFile = File(...)):
    # читаем байты
    contents = await file.read()

    # открываем через PIL
    image = Image.open(io.BytesIO(contents))

    # пример обработки
    width, height = image.size
    mode = image.mode

    # можно показать локально (если есть GUI)
    image.show()

    return "Сервер получил скриншот"