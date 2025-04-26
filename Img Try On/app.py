from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import torch
import cv2
import mediapipe as mp
from transformers import SamModel, SamProcessor
from torchvision import transforms
from diffusers.utils import load_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model once at startup
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

# Pose function
def get_shoulder_coordinates(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        height, width, _ = image.shape
        landmarks = results.pose_landmarks.landmark
        left_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
        right_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))
        return left_shoulder, right_shoulder
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        person_file = request.files['person_image']
        tshirt_file = request.files['tshirt_image']
        
        person_path = os.path.join(UPLOAD_FOLDER, 'person.jpg')
        tshirt_path = os.path.join(UPLOAD_FOLDER, 'tshirt.png')

        person_file.save(person_path)
        tshirt_file.save(tshirt_path)

        # Run your model
        coordinates = get_shoulder_coordinates(person_path)
        if coordinates is None:
            return "No pose detected."

        img = load_image(person_path)
        new_tshirt = load_image(tshirt_path)

        left_shoulder, right_shoulder = coordinates
        input_points = [[[left_shoulder[0], left_shoulder[1]], [right_shoulder[0], right_shoulder[1]]]]

        inputs = processor(img, input_points=input_points, return_tensors="pt")
        outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                              inputs["original_sizes"].cpu(),
                                                              inputs["reshaped_input_sizes"].cpu())

        mask_tensor = masks[0][0][2].to(dtype=torch.uint8)
        mask = transforms.ToPILImage()(mask_tensor * 255)

        new_tshirt = new_tshirt.resize(img.size, Image.LANCZOS)
        img_with_new_tshirt = Image.composite(new_tshirt, img, mask)

        result_path = os.path.join(OUTPUT_FOLDER, 'result.jpg')
        img_with_new_tshirt.save(result_path)

        return render_template('index.html', result_img='outputs/result.jpg')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
