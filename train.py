import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model, Sequential
# import section end ------------------------------------------------------
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TransferLearningClassifier:
    """
    1. Class for making and training the model on a given ```Image Classification``` dataset.
    
    2. It takes in train path,test path,input shape and labels as params.(__init__) 
    
    3. Builds the model(make_vgg16_model) and trains(train_model) it.
    
    4. Also provides functionality for visualization of dataset(explore_dataset) and training(plot_train)
    
    5. Can be used for any type of dataset or anytype of pretrained model.
    """
    def __init__(self, train_path:str, test_path:str,shape:list=[196,196],labels:int=7,model_file_name="model.h5"):
        """
        Constructor for class.
        Params:
            train_path(str): path to the training directory
            test_path(str): path to the testing directory
            shape(list): input shape for the model
            labels: no of labels in the dataset
        """
        self.shape = shape
        self.model_file = model_file_name
        self.train_path = train_path  # path to training data
        self.test_path = test_path  # path to testing/validation data
        self.labels = labels # no of labels
        self.train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            zoom_range=0.2,
                                            shear_range=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            width_shift_range=0.4,
                                            height_shift_range=0.4,
                                            brightness_range=[0.4, 1.0],
                                            rotation_range=40,
                                            validation_split=0.2)
        # Image generator object for training data

        
        # self.val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        # Image generator object for validation data

        self.train_data_gen = self.train_gen.flow_from_directory(self.train_path,
                                                                 target_size=self.shape,
                                                                 class_mode='categorical',
                                                                 batch_size=32,
                                                                 subset="training")

        self.val_data_gen = self.train_gen.flow_from_directory(self.train_path,
                                                               target_size=self.shape,
                                                               class_mode='categorical',
                                                               batch_size=32,
                                                               subset="validation")
        

        self.make_vgg16_model()# will store model object


    def explore_dataset(self):
        """
        Method for exploratory analysis of the data.
        Params:
        None
        """
        pass

    def make_vgg16_model(self):
        """
        Method to intialize vgg16 model with custom layers for fine tuning it.
        params:
        None
        """
        print("inside model building function")
        vgg = MobileNetV2(weights='imagenet',include_top=False,input_shape=tuple(self.shape+[3]))
        for layer in vgg.layers:
            layer.trainable = False #making all the layers non-trainable
    
        x = Flatten()(vgg.output) #flattening out the last layer
        predictions = Dense(self.labels,activation='softmax')(x) #Dense layer to predict wether their is pneumonia or not
        self.model = Model(inputs=vgg.input, outputs=predictions)# making the model
        self.model.summary() #model summary

    def train_model(self,epochs:int=20):
        """
        Method to train the model.
        params:
            epochs(int): no of epochs to train the model
        """
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history = self.model.fit_generator(self.train_data_gen,
                                 steps_per_epoch=200,
                                 epochs=epochs,
                                 validation_data=self.val_data_gen,
                                 validation_steps=100)
        self.plot_train(hist=history)
        self.model.save(self.model_file)

    def plot_train(self, hist):
        """
        Method to plot the training and validation performance curves of the model.
        Params:
        hist: keras history object for parsing the values.
        """
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(hist.history["loss"], label="train_loss")
        plt.plot(hist.history["val_loss"], label="val_loss")
        plt.plot(hist.history["accuracy"], label="train_acc")
        plt.plot(hist.history["val_accuracy"], label="val_acc")
        plt.title("Model Training")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("epochs.png")

train_path = r"G:\images/"
test_path = r"G:\images/"

tlc = TransferLearningClassifier(train_path=train_path,test_path=test_path,shape=[224,224],labels=5,model_file_name="mobile_net.h5")
tlc.train_model(2)