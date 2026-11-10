import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (150, 150)
BATCH_SIZE = 16

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory="test",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model = tf.keras.models.load_model("casting_defect_cnn.keras")

predictions = model.predict(test_generator, verbose=1)
pred_classes = (predictions >= 0.5).astype(int).flatten()

true_classes = test_generator.classes

precision = precision_score(true_classes, pred_classes)
recall = recall_score(true_classes, pred_classes)
f1 = f1_score(true_classes, pred_classes)
cm = confusion_matrix(true_classes, pred_classes)

print("\n========== EVALUATION METRICS ==========")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=list(test_generator.class_indices.keys())))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as 'confusion_matrix.png'")
