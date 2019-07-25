import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

cap_duration = 10
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('.', 'snapshots', 'head_detection_model_res50.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

print(model.summary())

names_to_labels = {'head':0}

labels_to_names = {v: k for k, v in names_to_labels.items()}


camera=cv2.VideoCapture(1)
while True:
    ret,image=camera.read()
    print(image.shape)
    # load image
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)


    print(boxes.shape)
    print(scores.shape)
    print(labels.shape)
    # correct for image scale
    boxes /= scale
    c=0
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        c+=1
        #color = label_color(label)

        
        #b = box.astype(int)
        #draw_box(draw, b, color=color)
        
        #caption = "{} {:.3f}".format(labels_to_names[label], score)
        #draw_caption(draw, b, caption)
    print("Present count:" ,c)  

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imshow('window',cv2.resize(draw,(640,480)))
    cv2.waitKey(1)

start_time = time.time()
#while(int(time.time() - start_time)< cap_duration):
#    cv2.VideoCapture(0)
