import os
import onnxruntime as ort
from keras_image_helper import create_preprocessor
import numpy as np


# train_transforms = transforms.Compose([
#     transforms.Resize((200, 200)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ) # ImageNet normalization
# ])

MODEL_NAME = os.getenv('MODEL_NAME', 'hair_classifier_empty.onnx')

def preprocess_pytorch_style(X):
    # X: shape (1, 299, 299, 3), dtype=floata32, values in [0,255]
    X = X / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    # convert NHWC -> NCHW
    # from (batch, height, width, channels) to (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std
    return X.astype(np.float32)

preprocessor = create_preprocessor(preprocess_pytorch_style, target_size=(200, 200))

# 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'

def predict(url):
    # img = download_image(url)
    # img = prepare_image(img, target_size=(200, 200))
    X = preprocessor.from_url(url)
    # X = np.expand_dims(X, axis=0)        # (1, 3, 200, 200)
    # X = X.unsqueeze(0).numpy()

    session = ort.InferenceSession(MODEL_NAME)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: X})
    return pred[0].item()

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }
    return result