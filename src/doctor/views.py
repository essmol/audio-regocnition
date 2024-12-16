import requests
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

def predict_image(request):
    if request.method == 'POST' and request.FILES:
        img_file = request.FILES['image']
        img_path = f'/tmp/{img_file.name}'
        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)

       
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        url = 'http://tensorflow_serving:8501/v1/models/xception_model:predict'
        json_data = {
            "signature_name": "serving_default",
            "instances": img_array.tolist()
        }
        response = requests.post(url, json=json_data)
        predictions = response.json()

        predicted_classes = decode_predictions(np.array(predictions['predictions']), top=5)[0]
        return JsonResponse(predicted_classes, safe=False)

    return render(request, 'upload.html')

