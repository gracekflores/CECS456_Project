import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def get_cam(model, image, last_conv_layer_name):
    grad_model = models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())

    # Normalize and plot
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    plt.imshow(image[0])
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.show()


def build_model(input_shape=(128,128,3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', 
        metrics=['accuracy', metrics.Precision(),
        metrics.Recall(),
        metrics.AUC(name='auc_roc')])
    
    return model

if __name__ == "__cnn_model__":
    # Define image dimensions and batch size
    IMG_WIDTH, IMG_HEIGHT = 128, 128
    BATCH_SIZE = 32
    # Set the paths to your main data directories
    TRAIN_DIR = 'chest_xray\\train'
    TEST_DIR = 'chest_xray\\test'
    VALIDATION_DIR = 'chest_xray\\val'
    
    # 1. Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=40,       # Data augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255) # Only normalization for testing

    validation_datagen = ImageDataGenerator(rescale=1./255) # Only normalization for validation

    # 2. Load data from directories
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    train_images, test_images = train_generator, test_generator
    validation_images = validation_generator

    model = build_model()
    model.summary()