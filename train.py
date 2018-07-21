# -*- coding: utf-8 -*-
"""
@author: akash
"""

import os
import numpy as np
import _pickle
import glob
import h5py
import matplotlib.pyplot as plt

from keras.models import Model
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns


class TrainClassifier():
    
    """
    Training Classifier class which uses state of the art deep learning model to extract features.
    Those extracted features are then fed to Logistic Regression Classifier which is used to predict the name of a person.
    """
    
    # Declaring final variables
    img_width, img_height = 224, 224
    train_data_dir = "./data/train"
    
    features_path = "./model/features.h5"
    labels_path = "./model/labels.h5"
    results = "./model/results.txt"
    classifier_path = "./model/classifier.cpickel"
    seed = 1994
    number_of_classes = 4
    
    
    def __init__(self):
        """
        Constructor initiates the training process.
        """
        self.loadBaseModel()
        self.extractFeatures()
        self.saveFeaturesLabels()
        #self.loadFeatures()
        self.splitDataset()
        self.trainClassifier()
        self.predict()
        self.saveClassifier()
        self.plot()
    
    def loadBaseModel(self):
        """
        Load the VGGFace module which is based on the VGG16 state of the art model.
        Using VGGFace for feature extraction. I am using the 5th block of Conv Layer to extract features from an Image.
        """
        #Load the base model 
        print("[INFO] Loading feature extractor")
        base_model = VGGFace(include_top=False, input_shape=(self.img_width, self.img_height, 3), pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('pool5').output)
        self.image_size = (self.img_width, self.img_height)
        
    
    def extractFeatures(self):
        """
        Loading train and test images from the data folder.
        Extracting features from every image obtained from the directory.
        Also using sklearn LabelEncoder for encoding the image labels.
        """
        # Label encoder to encode directory names (Eg. Cupcake becomes 0, Muffin becomes 1)
        train_labels = os.listdir(self.train_data_dir)
        le = LabelEncoder()
        le.fit([tl for tl in train_labels])
        
        # Extract features and store them in list
        print("[INFO] Started extracting features from the images")
        self.features = []
        self.labels   = []
        for label in train_labels:
        	cur_path = self.train_data_dir + "/" + label
        	for image_path in glob.glob(cur_path + "/*.jpg"):
        		img = image.load_img(image_path, target_size=self.image_size)
        		x = image.img_to_array(img)
        		x = np.expand_dims(x, axis=0)
        		x = utils.preprocess_input(x)
        		feature = self.model.predict(x)
        		flat = feature.flatten()
        		self.features.append(flat)
        		self.labels.append(label)
        	print("[INFO] Completed extracting features of - %s" % label)
    
        le = LabelEncoder()
        self.le_labels = le.fit_transform(self.labels)
        
        
    def saveFeaturesLabels(self):
        """
        Saving features and labels in h5py files, which can be used whenever needed.
        Saves feature extraction processing time.
        """
        print("[INFO] Saving image features in a file")
        h5f_data = h5py.File(self.features_path, 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(self.features))
        
        h5f_label = h5py.File(self.labels_path, 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(self.le_labels))
        
        h5f_data.close()
        h5f_label.close()
    
    
    def loadFeatures(self):
        """
        Load features and encoded labels from h5py files.
        """
        print("[INFO] Started loading image features from the file")
        h5f_data = h5py.File(self.features_path, 'r')
        h5f_label = h5py.File(self.labels_path, 'r')
        features_string = h5f_data['dataset_1']
        labels_string   = h5f_label['dataset_1']
        
        self.features = np.array(features_string)
        self.le_labels   = np.array(labels_string)
        
        h5f_data.close()
        h5f_label.close()
    
    
    def splitDataset(self):
        """
        Split the data into training and test set
        """
        # Split 80% of the data in training set and remaining 20% in test set 
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(np.array(self.features), 
                                                                  np.array(self.le_labels),
                                                                  test_size=0.2, 
                                                                  random_state=self.seed)
        
    
    def trainClassifier(self):
        """
        Using LogisticRegression classifier for this problem.
        Then we fit the classifier with training features and labels.
        """
        # Initialize classifier and fit the data
        print("[INFO] Started training classifier")
        self.classifier = LogisticRegression(random_state=self.seed)
        self.classifier.fit(self.X_train, self.y_train)
        
    
    def predict(self):
        """
        Check the result of the classifier on a testing set of images.
        Save the results of the classifier in the text file.
        """
        f = open(self.results, "w")
        # Evaluate the model of test data
        self.preds = self.classifier.predict(self.X_test)
        # Write the classification report to file
        f.write("{}\n".format(classification_report(self.y_test, self.preds)))
        f.close()
        
        
    def saveClassifier(self):
        """
        Save the classifier in pickled format
        """
        print("[INFO] Saving the classifier in a file")
        f = open(self.classifier_path, "wb")
        f.write(_pickle.dumps(self.classifier))
        f.close()


    def plot(self):
        """
        Plot the confusion matrix graph for visualization
        """
        labels = sorted(list(os.listdir(self.train_data_dir)))
        # plot the confusion matrix
        cm = confusion_matrix(self.y_test, self.preds)
        fig = plt.figure(figsize=(8, 8))
        plt.rcParams.update({'font.size': 20})
        sns.heatmap(cm, 
                    annot=True,
                    cmap="Set3")
        plt.show()
    
    

if __name__ == '__main__':
    t = TrainClassifier()