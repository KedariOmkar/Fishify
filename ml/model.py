import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_or_load_model(dataset_path, model_filename='freshness_detection_model.h5', img_width=150, img_height=150, epochs=10, batch_size=32):
    # Check if the model file already exists
    if os.path.exists(model_filename):
        print(f"Model {model_filename} already exists. Loading the existing model.")
        model = tf.keras.models.load_model(model_filename)
    else:
        print("Training the model.")
        model = train_model(dataset_path, img_width, img_height, epochs, batch_size)
        model.save(model_filename)

    return model
def train_model(dataset_path, img_width=150, img_height=150, epochs=10, batch_size=32):
    # Data Augmentation for better generalization
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Load and augment the training dataset
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    # Load and augment the validation dataset
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Build a simple CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Save the trained model
    model.save('freshness_detection_model.h5')

def predict_image(model, image_path, img_width=150, img_height=150):
    # Load and preprocess the input image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    return predictions[0][0]

def visualize_prediction(image_path, prediction_score):
    # Display the prediction result
    print(f"Prediction Score: {prediction_score}")

    if prediction_score > 0.5:
        print("Prediction: Spoiled")
    else:
        print("Prediction: Fresh")

# Example usage:
model = train_or_load_model('ml/dataset')
prediction_score = predict_image(model, "D:\\_MG_9140.JPG")
visualize_prediction("D:\\_MG_9140.JPG", prediction_score)
