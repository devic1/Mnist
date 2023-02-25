#importing neccessary libraries
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template

#creating a new instance of the Flask web application.
app = Flask(__name__)

#loading the CNN model architecture
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

#Processing device (cuda or cpu)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Initializing the Model
model = Net().to(device)

#loading weights with load_state_dict
model.load_state_dict(torch.load('mnist_cnn.pt',map_location=device))

#entering into eval mode to avoid gradient tracking
model.eval()


"""compiling transforms like -
	PIL Image to tensor
	Resizing to 28 X 28
	converting it to GrayScale
	Normalizing image with mean and standrad deviation
	adding extra dimensions for considering as a batch"""

def transform_image(image_bytes):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mean_value = cv2.mean(image_array)[0]
    _, thresholded = cv2.threshold(image_array, mean_value, 255, cv2.THRESH_BINARY)
    thresholded_image = Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
    t = transform(thresholded_image)
    return t.unsqueeze(0)

#getting prediction by forwarding it to the model and taking the maximum elements index 
def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        tensor = tensor.to(device)
        outputs = model.forward(tensor)
        return outputs.argmax().item()
    except:
        print("Some Error Occured")
        return 7

#returns index.html template 
@app.route('/')
def hello():
    return render_template("index.html")

#returns result.html
@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['image']
    t = file.read()
    #checks to predict image or board
    if t != b'':
        res = get_prediction(t)
    else:
        change = request.form['pixelcount']
        li = change.split(' ')[:-1]
        pix = list(map(int,li))
        N = torch.zeros((784,1),device=device)
        for i in pix:
            N[i] = 255
        N = N.view(1,1,28,28)
        out = model(N)
        res = out.argmax().item()
    return render_template("result.html",message=res,ico="images/"+str(res)+".ico")

#starts the flask development server
if __name__ == '__main__':
    app.run()
