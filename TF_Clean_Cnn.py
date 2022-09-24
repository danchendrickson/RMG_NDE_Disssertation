# hat tip: https://developers.google.com/codelabs/tensorflow-5-compleximages#0

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

im_h = 500
im_w = 10000

run_name = 'TF_BigData'

batchSize = 6

#tf.config.set_visible_devices([], 'GPU')

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (6,6), activation='relu', input_shape=(im_h, im_w, 3)),
    tf.keras.layers.MaxPooling2D(4, 4),
    # The second convolution
    tf.keras.layers.Conv2D(32, (4,4), activation='relu'),
    tf.keras.layers.MaxPooling2D(4,4),
    # The third convolution
    tf.keras.layers.Conv2D(32, (4,4), activation='relu'),
    tf.keras.layers.MaxPooling2D(4,4),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.05),
              metrics=['acc'])

print('Big Data folders')

imFolder = '/sciclone/scr10/dchendrickson01/BigData/'

# All images will be rescaled by 1./255
generator = ImageDataGenerator(rescale=1./255, validation_split = 0.2)
 
# Flow training images in batches of 128 using train_datagen generator
train_generator = generator.flow_from_directory(
        imFolder,  # This is the source directory for training images
        target_size=(im_h, im_w),  # All images will be resized to 150x150
        batch_size=batchSize,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary',
        subset='training')
validation_generator = generator.flow_from_directory(
        imFolder,  # This is the source directory for training images
        target_size=(im_h, im_w),  # All images will be resized to 150x150
        batch_size=batchSize,
        class_mode='binary',
        subset='validation')

StepsNeeded=len(train_generator)

history = model.fit(
      train_generator,
      steps_per_epoch=StepsNeeded,  
      epochs=9,
      validation_data=validation_generator,
      validation_steps=len(validation_generator)/4,
      verbose=1)


model.save(imFolder+run_name+'/')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(imFolder + 'ModelAccuracy'+run_name+'.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(imFolder + 'ModelLoss'+run_name+'.png')
plt.show()