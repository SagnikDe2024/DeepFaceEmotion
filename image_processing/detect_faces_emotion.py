from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
from ultralytics import YOLO


class DetectFaces:
	def __init__(self):
		print(f'Current working directory is {Path.cwd()}')
		self.model: YOLO = YOLO('yolov8n-face.pt')
		self.image: Optional['Image'] = None
		self.inference_results = []

	def upload(self, file: BytesIO):
		self.image = Image.open(file).convert('RGB')
		self.inference_results = []

	def draw_bbox_with_emotion(self, io_location: BytesIO):
		draw = ImageDraw.Draw(self.image)
		fnt = ImageFont.truetype("GlacialIndifference-Regular.otf", 16)

		for bbox in self.inference_results:
			x1 = int(bbox['x1'])
			y1 = int(bbox['y1'])
			x2 = int(bbox['x2'])
			y2 = int(bbox['y2'])
			draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 85), width=1)
			emotion = f"{bbox['emotion']}, confidence={int(bbox['emo_confidence']*100)}%"
			draw.text((x1, y2), emotion, font=fnt, fill=(128, 0, 128))

		emotion_results = [{'index': res['f_i'], 'emotion': res['emotion'], 'emo_confidence': res['emo_confidence']}
						   for
						   res in self.inference_results]
		self.image.save(io_location, 'PNG')
		return emotion_results

	def detect_faces(self):
		inference_results = []
		inferences: list = self.model(self.image)
		detected_faces = 0
		for image_number, inference in enumerate(inferences):
			for i, detected_face in enumerate(inference):
				given_xyxy = detected_face.boxes.xyxy[0]
				x1 = int(given_xyxy[0].item())
				y1 = int(given_xyxy[1].item())
				x2 = int(given_xyxy[2].item())
				y2 = int(given_xyxy[3].item())
				h, w = inference.orig_shape
				face_box = inference.orig_img[y1:y2, x1:x2]
				emotional_resp = get_emotion(face_box)
				common_file_data = {'h': h, 'w': w, 'f_i': i}
				xys = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
				all_facial_results = {**common_file_data, **xys, **emotional_resp}
				inference_results.append(all_facial_results)
				detected_faces += 1
		self.inference_results = inference_results
		print(f'inference_results {self.inference_results}')
		return detected_faces


def get_emotion(cropped_face):
	analyzed_emotion = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False)[0]
	return {'emotion': analyzed_emotion['dominant_emotion'], 'emo_confidence': analyzed_emotion['face_confidence']}
