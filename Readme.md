# DeepFaceEmotion

## Project Summary

DeepFaceEmotion is a web application that analyzes emotions in faces from uploaded images. Through an easy-to-use web
interface, users can upload images, and the application will:

1. Automatically detect all faces in the image
2. Analyze the emotional expression of each detected face (such as happy, sad, angry, etc.)
3. Display the processed image with highlighted faces and their emotional states
4. Provide confidence scores for the detected emotions

The application is built as a FastAPI web service, making it suitable for both personal use and integration into larger
systems through its API endpoints.

1. Clone the repository

```bash git clone https://github.com/yourusername/DeepFaceEmotion.git```

```cd DeepFaceEmotion```

3. Create and activate a Conda environment:

```
bash conda create -n DeepFaceEmotionEnv python=3.12 conda activate DeepFaceEmotionEnv
``` 

3. Install the required dependencies:

```
bash pip install -r requirements.txt
``` 

## Project Structure

```
DeepFaceEmotion/
 ├── image_processing/ # Image processing related modules 
 ├── templates/ # Web interface templates 
 ├── main.py # Main application file 
 ├── detect_faces_emotion.py # Core detection and emotion analysis logic 
 ├── requirements.txt # Project dependencies 
 └── yolov8n-face.pt # Pre-trained YOLO model for face detection
``` 

## Usage

1. Make sure your Conda environment is activated:

```
conda activate DeepFaceEmotionEnv
``` 

2. Run the application:

```
uvicorn main:app
``` 

3. Access the web interface through your browser (the address will be shown in the console when you run the application)

4. Upload an image through the interface to detect faces and analyze emotions

## Output

The system will:

- Detect faces in the uploaded image
- Analyze the emotion of each detected face
- Draw bounding boxes around the faces
- Display emotion labels with confidence scores
- Return the processed image with visual indicators

## Note

This project requires the following model files:

- `yolov8n-face.pt` (YOLO model for face detection)
- `GlacialIndifference-Regular.otf` (Font file for text rendering)

Make sure these files are present in the root directory of the project.

## License

No idea ... this is a private project. Check the license of the used libraries.
