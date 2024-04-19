import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='binary')

# Model Building
base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                         include_top=False,
                                         weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
model.fit(train_set, epochs=10, validation_data=test_set)

# Model Evaluation
loss, accuracy = model.evaluate(test_set)
print(f'Test Accuracy: {accuracy}')

# Save the model
model.save('cat_dog_classifier.h5')
