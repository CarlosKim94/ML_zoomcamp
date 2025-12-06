FROM agrigorev/model-2025-hairstyle:v1

RUN pip install onnxruntime keras_image_helper

COPY homework.py .

ENV MDDEL_NAME=/var/task/hair_classifier_empty.onnx

CMD ["homework.lambda_handler"]