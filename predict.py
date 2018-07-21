# -*- coding: utf-8 -*-
"""
@author: akash
"""

import os
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import _pickle

from keras_vggface.vggface import VGGFace
from keras_vggface import utils

from attendance import Attendance

class PredictImage:
    
    
    img_width, img_height = 224, 224
    train_data_dir = "./data/train"
    classifier_path = "./model/classifier.cpickel"
    att = Attendance()
    
    def __init__(self):
        """
        Constructor loads the model and a classifier. It sets everything up for image prediction.
        """
        self.train_labels = sorted(list(os.listdir(self.train_data_dir)))
        self.loadModel()
        self.loadClassifier()
    
    def loadModel(self):
        """
        Load the VGGFace model which is used to extract features from the images.
        """
        base_model = VGGFace(include_top=False, input_shape=(self.img_width, self.img_height, 3), pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('pool5').output)
        self.image_size = (self.img_width, self.img_height)
    
    
    def loadClassifier(self):
        """
        Load the classifier from the saved pickle file.
        """
        self.classifier = _pickle.load(open(self.classifier_path, 'rb'))
    
    
    def predictImg(self, path):
        """
        Convert the image into an array for processing.
        """
        img = image.load_img(path, target_size=self.image_size)
        x = image.img_to_array(img)
        return self.predict(x)
    
    
    def predict(self, x):
        """
        Predict the class of the image array and return the name of a class
        """
        x = np.expand_dims(x, axis=0)
        #x = utils.preprocess_input(x, version=1)
        feature = self.model.predict(x)
        flat = feature.flatten()
        flat = np.expand_dims(flat, axis=0)
        preds = self.classifier.predict(flat)
        prediction = self.train_labels[preds[0]]
        self.att.markAttendance(prediction)
        return prediction


if __name__ == '__main__':
    pass
    #p = PredictImage()
    #print(p.predictImg('./data/predict/11.jpg'))