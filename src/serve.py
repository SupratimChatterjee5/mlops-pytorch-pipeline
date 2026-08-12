import io
import os
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from model import get_model

app=Flask(__name__)
MODEL_PATH=Path(os.getenv("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"))
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=None

def load_inference_model():
    global model
    if MODEL_PATH.exists():
        model=get_model(architecture="resnet18",num_classes=10)
        checkpoint=torch.load(MODEL_PATH,map_location=DEVICE)
        state_dict=checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

load_inference_model()

transform =transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914,0.4822,0.4465],
        std=[0.2470, 0.2435, 0.2616] #Values for CIFAR-10 dataset
    )
])

@app.route("/health",methods=["GET"])
def health():
    if model is not None:
        return jsonify({"status":"healthy", "model_loaded":True}), 200
    return jsonify({"status":"unhealthy", "model_loaded":False}), 500

@app.route("/predict",methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model non-functional/unloaded"}), 500
    if "image" not in request.files:
        return jsonify({"error": "No image field provided"}), 400

    files=request.files["image"]
    image_bytes=files.read()
    image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor=transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs=model(tensor)
        probs=torch.softmax(outputs,dim=1).squeeze().tolist()

    classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    predictions={classes[i]: round(probs[i],4) for i in range(len(classes))}

    return jsonify({
        "predictions": predictions,
        "predicted_class": classes[int(torch.argmax(outputs))]
    }), 200

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)