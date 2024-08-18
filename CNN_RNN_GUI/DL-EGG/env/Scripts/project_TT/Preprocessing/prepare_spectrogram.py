from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to image directories
healthy_images_path = 'data/healthy_images/'
seizures_images_path = 'data/seizures_images/'

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    directory='data/',
    target_size=(128, 128),  # Resize images to the desired size
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory='data/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
