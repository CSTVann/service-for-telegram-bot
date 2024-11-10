import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,MaxPool2D,concatenate
from tensorflow.keras.layers import Conv2D,  concatenate, GlobalAveragePooling2D
from PIL import Image
import warnings
import numpy as np



labels = ['Anthracnose','Downy_Mildew','Bacterial_Wilt','Gummy_Stem_Bligh','Fresh_Leaf']
class_mapping = dict(zip(list(range(5)),labels))

def inception_module(x, filters):
    # 1x1 Convolution for dimensionality reduction
    conv1x1_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    # 3x3 Convolution
    conv3x3 = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)

    # 5x5 Convolution
    conv5x5 = Conv2D(filters[2], (5, 5), padding='same', activation='relu')(x)

    # MaxPooling
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    conv1x1_2 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate the outputs along the channel axis
    inception_output = concatenate([conv1x1_1, conv3x3, conv5x5, conv1x1_2], axis=-1)
    return inception_output
    
def model() : 
	# Input layer
	Input = tf.keras.Input(shape=(256, 256, 3))

	# Rest of the layers
	conv1 = Conv2D(filters=96, kernel_size=11, dilation_rate=2)(Input)
	pool1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv1)
	conv2 = Conv2D(filters=128, kernel_size=5, dilation_rate=2)(pool1)
	pool2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv2)
	conv3 = Conv2D(filters=192, kernel_size=3, dilation_rate=2)(pool2)
	pool3 = MaxPool2D(pool_size=(2, 2), strides=2)(conv3)
	conv4 = Conv2D(filters=192, kernel_size=3, dilation_rate=2)(pool3)
	pool4 = MaxPool2D(pool_size=(2, 2), strides=2)(conv4)

	# Inception module
	inception_output = inception_module(pool4, filters=[64, 128, 32, 32])

	# Final Convolutional Layer
	conv5 = Conv2D(filters=128, kernel_size=3, dilation_rate=2)(inception_output)

	# Global Average Pooling
	global_avg_pooling = GlobalAveragePooling2D()(conv5)

	# Classification layer (with a different name than 'Dense')
	output_layer = tf.keras.layers.Dense(5, activation='softmax')(global_avg_pooling)

	# Create the model
	model = tf.keras.Model(inputs=Input, outputs=output_layer)

	# Compile the model and specify the loss function, optimizer, and metrics as needed
	model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
	
	return model
	
	

class GPDCNN() : 
	def __init__(self):
		self.model = model()
		self.is_weights_initialized = False
		self.weights_url = 'https://github.com/ali0salimi/cucumber-leaf-disease-prediction/raw/main/model_weights.hdf5'
		#self.weights_url = 'model_weights.hdf5'
	
	def load_weights(self,weights_path) : 
		self.model.load_weights(path)
		
	def preprocess_input(self,img) :
		return np.array(img.resize((256,256),resample=Image.BILINEAR))[:,:,:3].reshape((1,256,256,3))/255
		
	def predict(self,image_path) :
		if not self.is_weights_initialized : 
			model_weights_path = tf.keras.utils.get_file('model_weights.hdf5', self.weights_url)
			self.model.load_weights(model_weights_path)
			self.is_weights_initialized = True
			# Disable all warnings
			warnings.filterwarnings("ignore")
		
		try:
			prediction = self.model.predict(self.preprocess_input(Image.open(image_path)))[0]
			predicted_disease = class_mapping[np.argmax(prediction)]
			confidence = float(np.max(prediction))  # Convert to Python float
			print(f"Predicted Disease: {predicted_disease}")
			print(f"Confidence: {confidence:.2f}")
			
			return predicted_disease, confidence
		except Exception as e:
			print(f"Error in prediction: {str(e)}")
			return None
