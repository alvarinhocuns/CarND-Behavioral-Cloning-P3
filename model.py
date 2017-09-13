import csv
import cv2
import numpy as np

lines =[]
with open("driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
steerCorr=[0.0,0.2,-0.2]

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if "steering" in batch_sample[3]: continue
                for i in range(3):
                    name = './IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    measurement = float(batch_sample[3])+steerCorr[i]
                    images.append( image )
                    angles.append( measurement )
                    #augmenting samples
                    images.append( cv2.flip(image,1) )
                    angles.append( measurement*(-1.0) )
                
    
            # trim image to only see section with road
            X_train = np.array(images)
            Y_train = np.array(angles)
            Y_train_onehot = np_utils.to_categorical(Y_train,10)
            yield shuffle(X_train, Y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
ch, row, col = 3, 160, 320  # Trimmed image format


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt


#Model definition
model = Sequential()
#Normalization layer
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(row,col,ch)))

#Crop layer to remove the front of the car and the horizont. Output shape=(3,65,320)
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(3,160,320),name='CropImage'))

#1st convolutionn layer. input shape=(3,65,320). output shape=(16,17,80). Border mode="Valid"
model.add(Convolution2D(16,8,8,subsample=(4,4), border_mode="same", activation="relu", name="ConvLayer1"))

#2nd convolutionn layer. input shape=(16,15,79). output shape=(32,9,40). Border mode="Valid"
model.add(Convolution2D(32,5,5,subsample=(2,2), border_mode="same", activation="relu", name="ConvLayer2"))

#3rd convolutionn layer. input shape=(32,6,38). output shape=(64,5,20). Border mode="Valid"
model.add(Convolution2D(64,5,5,subsample=(2,2), border_mode="same", activation="relu", name="ConvLayer3"))

model.add(Flatten( name="Flatten" ))
model.add(Dropout(0.5, name="Dropout_0.5" ))
model.add(ELU( name="Activation1_ELU" ))
model.add(Dense(100, name="FullyConnected1" ))
model.add(ELU(name="Activation2_ELU" ))
model.add(Dense(1, name="Output_Layer" ))


model.compile(loss="mse",optimizer="adam")
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3*2, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3*2, nb_epoch=6)

#Save the model
model.save("model.h5")

#Save the model in graphical format
from keras.utils.visualize_util import plot
plot(model, to_file='model.png',show_shapes=True)

#printing mean squared errors
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

