from io import BytesIO

from fastapi import FastAPI, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from image_processing.detect_faces_emotion import DetectFaces

app = FastAPI()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

savedFiles: dict[str, BytesIO] = {}
bounding_boxes: dict[str, list] = {}
templates = Jinja2Templates(directory="templates")
detect_faces = DetectFaces()


def allowed_file(filename: str):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
	return templates.TemplateResponse("upload.html", {"request": request, "image_list": list(savedFiles.keys())})


@app.post("/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile):

	if file and allowed_file(file.filename):
		filename = file.filename
		file_obj = BytesIO(await file.read())
		file_obj.seek(0)

		detect_faces.upload(file_obj)
		detect_faces.detect_faces()

		image_data = BytesIO()

		face_bounding_boxes = detect_faces.draw_bbox_with_emotion(image_data)
		savedFiles[filename] = image_data
		bounding_boxes[filename] = face_bounding_boxes

		return templates.TemplateResponse("upload.html",
				{"request": request, "filename": filename, "image_list": list(savedFiles.keys())})
	else:
		return templates.TemplateResponse("upload.html", {"request": request, "error": "File type not allowed"})


@app.get("/bbox/{filename}")
async def get_emotions(filename: str):
	detected = detect_faces.detect_faces()
	print(f'Faces detected {detected}')

	if filename not in bounding_boxes.keys():
		return {"error": "No bounding box for image"}, 404

	emotion = bounding_boxes[filename]

	return emotion


@app.get("/image/{filename}")
async def get_image(filename: str):

	if filename not in savedFiles.keys():
		return {"error": "Image not found"}, 404

	image_data = savedFiles[filename]
	image_data.seek(0)

	return StreamingResponse(image_data, media_type="image/*")
