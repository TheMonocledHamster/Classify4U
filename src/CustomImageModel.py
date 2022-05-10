from tensorflow import keras
import numpy as np
import json
import cv2 as opencv

class CustomImageClassifier:
    """
     The following functions are required to be called before a classification can be made
    * setModelPath() , path to your custom model
    * setJsonPath , , path to your custom model's corresponding JSON file
    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsMobileNetV2(), setModelTypeAsResNet50(), setModelTypeAsDenseNet121, setModelTypeAsInceptionV3]
    * loadModel() [This must be called once only before making a classification]
    Once the above functions have been called, you can call the predictImages() function of the classification instance
    object at anytime to predict an image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.jsonPath = ""
        self.batchSize = 4
        self.__model_classes = dict()
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_size = 224
    
    def setModelPath(self, modelPath):
        """
        'setModelPath()' function is required and is used to set the file path to the custom model adopted from the list of the
        available model types. The model path must correspond to the model type set for the classification instance object.
        :param model_path:
        :return:
        """
        self.modelPath = modelPath
    
    def setJsonPath(self, model_json):
        """
        'setJsonPath()' function is required and is used to set the file path to the json file corresponding to the model.
        :param model_path:
        :return:
        """
        self.jsonPath = model_json
    
    def setBatchSize(self, batchSize):
        """
        'setBatchSize()' function is used to set the batch size for the prediction.
        :param batchSize:
        :return:
        """
        self.batchSize = batchSize
    
    def setModelTypeasResnet50(self):
        """
        'setModelTypeAsResNet50()' is used to set the model type to the ResNet50 model
                for the classification instance object .
        :return:
        """
        self.__modelType = 'resnet50'

    def setModelTypeAsMobileNetV2(self):
        """
        'setModelTypeAsMobileNetV2()' is used to set the model type to the MobileNetV2 model
        for the classification instance object.
        :return:
        """
        self.__modelType = 'mobilenetv2'
    
    def setModelTypeAsDenseNet121(self):
        """
         'setModelTypeAsDenseNet121()' is used to set the model type to the DenseNet121 model
                for the classification instance object.
        :return:
        """
        self.__modelType = 'densenet121'

    def setModelTypeasInceptionV3(self):
        """
        'setModelTypeAsInceptionV3()' is used to set the model type to the InceptionV3 model
                for the classification instance object.
        :return:
        """
        self.__modelType = 'inceptionv3'
    
    def loadModel(self,prediction_speed="normal",num_objects=1000):
        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. This function receives an optional value which is "prediction_speed".
        The value is used to reduce the time it takes to classify an image, down to about 50% of the normal time,
        with just slight changes or drop in classification accuracy, depending on the nature of the image.
        * prediction_speed (optional); Acceptable values are "normal", "fast", "faster" and "fastest"
        * num_objects (optional); the number of objects the model is trained to recognize
        :param prediction_speed:
        :param num_objects:
        :return:
        """
        self.__model_classes = json.load(open(self.jsonPath))

        if(prediction_speed=="normal"):
            self.__input_image_size = 224
        elif(prediction_speed=="fast"):
            self.__input_image_size = 160
        elif(prediction_speed=="faster"):
            self.__input_image_size = 120
        elif(prediction_speed=="fastest"):
            self.__input_image_size = 100
        else:
            raise ValueError("Invalid classification speed value")
        
        if(self.__modelLoaded == False):
            if(self.__modelType == ""):
                raise ValueError("Model type not set")
            elif(self.__modelType == "resnet50"):
                try:
                    model= keras.applications.ResNet50(input_shape=(self.__input_image_size, self.__input_image_size, 3),weights=self.modelPath,classes=num_objects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a ResNet50 Model and is located in the path {}".format(self.modelPath))
            elif(self.__modelType == "mobilenetv2"):
                try:
                    model= keras.applications.MobileNetV2(input_shape=(self.__input_image_size, self.__input_image_size, 3),weights=self.modelPath,classes=num_objects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a MobileNetV2 Model and is located in the path {}".format(self.modelPath))
            elif(self.__modelType == "densenet121"):
                try:
                    model= keras.applications.DenseNet121(input_shape=(self.__input_image_size, self.__input_image_size, 3),weights=self.modelPath,classes=num_objects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a DenseNet121 Model and is located in the path {}".format(self.modelPath))
            elif(self.__modelType == "inceptionv3"):
                try:
                    model= keras.applications.InceptionV3(input_shape=(self.__input_image_size, self.__input_image_size, 3),weights=self.modelPath,classes=num_objects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("An error occured. Ensure your model file is a InceptionV3 Model and is located in the path {}".format(self.modelPath))
            else:
                raise ValueError("Invalid model type, please utilize one of the available model types")
        else:
            raise ValueError("Model already loaded")
    
    def loadFullModel(self, prediction_speed="normal", num_objects=10):
        """
        'loadFullModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. As opposed to the 'loadModel()' function, you don't need to specify the model type.
        - prediction_speed (optional), Acceptable values are "normal", "fast", "faster" and "fastest"
        - num_objects (required), the number of objects the model is trained to recognize
        
        :param prediction_speed:
        :param num_objects:
        :return:
        """

        self.numObjects = num_objects
        self.__model_classes = json.load(open(self.jsonPath))

        if (prediction_speed == "normal"):
            self.__input_image_size = 224
        elif (prediction_speed == "fast"):
            self.__input_image_size = 160
        elif (prediction_speed == "faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):
            model = keras.models.load_model(filepath=self.modelPath)
            self.__model_collection.append(model)
            self.__modelLoaded = True
            self.__modelType = "custom"
    
    def getModels(self):
        """
        'getModels()' provides access to the internal model collection. Helpful if models are used down the line with tools like lime.
        :return:
        """
        return self.__model_collection

    def predictImages(self, image_input, input_type="file", batchSize=1):
        """
        'predictImage()' function is used to predict the class of a given image/batch of images by receiving the following arguments:
            * image_input , file path/numpy array/image file stream of the images.
            * input_type (optional) , the type of input to be parsed. Acceptable values are "file", "array" and "stream"
            * result_count (optional) , the number of classifications to be sent which must be whole numbers between
                1 and 1000. The default is 5.
        This function returns 2 arrays namely 'prediction_result' and 'prediction_probability', containing top predictions and 
        corresponding likelihoods respectively for the batch.
        
        :param input_type:
        :param image_input:
        :return prediction_result, prediction_probability:
        """

        if(self.__modelLoaded == False):
            raise ValueError("Model not loaded")
    
        if(input_type == "file"):
            try:
                images_to_predict = keras.preprocessing.image.load_img(image_input, target_size=(self.__input_image_size, self.__input_image_size))
                images_to_predict = keras.preprocessing.image.img_to_array(images_to_predict, data_format="channels_last")
                images_to_predict = np.expand_dims(images_to_predict, axis=0)
            except:
                raise ValueError("An error occured. Ensure your image is located in the path {}".format(image_input))
        elif(input_type == "array"):
            try:
                # images_to_predict = np.resize(image_input, (batchSize, self.__input_image_size, self.__input_image_size, 3))
                images_to_predict = opencv.resize(image_input, (batchSize, self.__input_image_size, self.__input_image_size, 3), interpolation=opencv.INTER_NEAREST)
            except:
                raise ValueError("An error occured. Your numpy array is not in the correct format")
        elif(input_type == "stream"):
            try:
                images_to_predict = opencv.resize(image_input, (batchSize, self.__input_image_size, self.__input_image_size, 3), interpolation=opencv.INTER_NEAREST)
            except:
                raise ValueError("An error occured. Your image stream is not in the correct format")
        
        if(self.__modelType == "mobilenetv2"):
            images_to_predict = keras.applications.mobilenet_v2.preprocess_input(images_to_predict)
        elif(self.__modelType == "densenet121"):
            images_to_predict = keras.applications.densenet.preprocess_input(images_to_predict)
        elif(self.__modelType == "inceptionv3"):
            images_to_predict = keras.applications.inception_v3.preprocess_input(images_to_predict)
        elif(self.__modelType == "resnet50"):
            images_to_predict = keras.applications.resnet50.preprocess_input(images_to_predict)
        elif(self.__modelType == "custom"):
            images_to_predict = keras.applications.mobilenet_v2.preprocess_input(images_to_predict)
        
        try:
            model = self.__model_collection[0]
            predictions = model.predict(images_to_predict, batch_size = self.batchSize)
            prediction_result = [self.__model_classes[str(np.argmax(prediction))] for prediction in predictions]
            prediction_probability = [prediction[np.argmax(prediction)] for prediction in predictions]
            return prediction_result, prediction_probability
        except:
            raise ValueError("Ensure your model is trained to recognize the objects you are trying to predict")
