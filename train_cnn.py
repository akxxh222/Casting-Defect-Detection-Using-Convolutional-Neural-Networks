import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 15

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_generator = train_datagen.flow_from_directory(
    directory="train",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    directory="test",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

for split_name, generator in [("Train", train_generator), ("Test", test_generator)]:
    print(f"\n{split_name} image counts:")
    for class_name, class_index in generator.class_indices.items():
        count = int((generator.classes == class_index).sum())
        print(f"  {class_name}: {count}")

model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), activation="relu",
                 input_shape=(150, 150, 1)))
model.add(MaxPooling2D(2, 2))

# Block 2
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

# Block 3
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

# Classification head
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

model.save("casting_defect_cnn.keras")
print("Model saved as casting_defect_cnn.keras")
