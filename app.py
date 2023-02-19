import io
import json
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt',map_location=torch.device("cpu")))
model.eval()

def transform_image(image_bytes):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    image = Image.open(io.BytesIO(image_bytes))
    t = transform(image)
    return t.unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    return outputs.argmax().item()

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['image']
    img_bytes = file.read()
    res = get_prediction(img_bytes)
    return render_template("result.html",message=res)


if __name__ == '__main__':
    app.run()
