# Attendance-using-facial-recognition

## Demo

![Alt Text](https://github.com/aakashrai1/Attendance-using-facial-recognition/blob/master/demo/out.gif)


##Background

The idea behind this project is to use AI to detect a person's face and mark his/her attendance.
(I expect this to be used heavily in the future)

The dataset used for this project are images clicked from my phone within the campus.
(For the personal reasons, I have not checked in the dataset images.)

For this project, I have used Keras-VGGFace module which is based on top of the VGG16 state of the art deep CNN model.
Keras-VGGFace is used to extract the facial features from the image, which eventually is used to train a Classifier.

## Dependencies

- Keras
- Keras-VGGFace
- sklearn
- numpy
- h5py
- matplotlib
- seaborn
- cv2

## Steps to train and run the code

1. Save the training images ordered in classes inside data/train folder.
2. Run the following command inside project directory
```sh
$ python train.py
```
3. Once the training completes, run
```sh
$ python face_detection.py
```
The command will start the webcam and it will start detecting faces.

## Some stats

- The dataset contained 390 images distributed in 4 classes. Out of which 80% was used for training and remaining for validation
- The Classifier achieved 96% accuracy for the test set.

![image](/screenshots/Report.png)
> Report

![image](/screenshots/Report.png)
> Confusion Matrix


## Mistakes I did

1) I tried inceptionv3 model by Google, removed the final layer and customized the fully connected layer for my use case.
I was under a wrong impression that, it's good enough to detect features from my different dataset.

2) I tried avoiding overfitting by dropping some of the neurons from the fully connected layer.
3) I wasted a lot of time by tweaking and training the model by customizing it for my use case.
4) Eventually, I realized the problem was not overfitting but underfitting. The number of images per class was far too less to train a deep CNN model.

I learned it the hard way by trying and testing by tweaking many hyper-parameters and the model itself.

Hope you avoid the mistakes I did.

Thank you.
