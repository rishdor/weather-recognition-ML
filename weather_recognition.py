import glob as gl
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import PIL

image_count = len(list(gl.glob('processed_dataset/**/*.jpg')+ gl.glob('processed_dataset/**/*.jpeg')))
print(f'{image_count} examples of weather')

categories = ['cloudy','foggy', 'lightning', 'rainy', 'snow', 'sunny', 'sunrise']

for category in categories:
    count = len(list(gl.glob(f'processed_dataset/{category}/*.jpeg')+ gl.glob(f'processed_dataset/{category}/*.jpg')))
    print(f"{category} count = {count}")

batch_size = 16
class_count = 7

img_height = 200
img_width = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
    'processed_dataset/',
    subset = 'training',
    validation_split = 0.1,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'processed_dataset/',
    subset = 'validation',
    validation_split = 0.1,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(f'class names: {class_names}')

train_ds = train_ds.cache().shuffle(3500).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(class_count, activation='softmax')
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
model.summary()

epochs=20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_validation.png')
# plt.show()

from sklearn.metrics import classification_report

def evaluate_model(val_ds,model):
    y_pred=[]
    y_true=[]

    for batch_images,batch_labels in val_ds:
        predictions=model.predict(batch_images,verbose=0)
        y_pred=y_pred+np.argmax(tf.nn.softmax(predictions),axis=1).tolist()
        y_true=y_true+batch_labels.numpy().tolist()
    print(classification_report(y_true,y_pred))

evaluate_model(val_ds,model) 
