import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar y preprocesar CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizar las imágenes
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir etiquetas a one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Funciones para construir los modelos
def build_cnn(input_shape, num_classes):
    """Modelo CNN modificado."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_efficientnetb0(input_shape, num_classes):
    """Modelo EfficientNetB0."""
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)

def build_mobilenetv2(input_shape, num_classes):
    """Modelo MobileNetV2."""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)

def build_squeezenet(input_shape, num_classes):
    """Modelo SqueezeNet."""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(128, (3, 3), activation='relu'),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_resnet20(input_shape, num_classes):
    """Modelo ResNet20."""
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)

def build_resnet50(input_shape, num_classes):
    """Modelo ResNet50 preentrenado."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)

# 3. Entrenar y evaluar cada modelo
models = {
    "CNN": build_cnn((32, 32, 3), 10),
    "EfficientNetB0": build_efficientnetb0((128, 128, 3), 10),
    "MobileNetV2": build_mobilenetv2((128, 128, 3), 10),
    "SqueezeNet": build_squeezenet((32, 32, 3), 10),
    "ResNet20": build_resnet20((32, 32, 3), 10),
    "ResNet50": build_resnet50((224, 224, 3), 10)  # Requiere redimensionar las imágenes
}

results = {}

for model_name, model in models.items():
    print(f"\nEntrenando modelo: {model_name}\n")
    
    # Redimensionar imágenes si es necesario
    if model_name in ["EfficientNetB0", "MobileNetV2"]:
        x_train_resized = tf.image.resize(x_train, (128, 128))
        x_test_resized = tf.image.resize(x_test, (128, 128))
    elif model_name == "ResNet50":
        x_train_resized = tf.image.resize(x_train, (224, 224))
        x_test_resized = tf.image.resize(x_test, (224, 224))
    else:
        x_train_resized, x_test_resized = x_train, x_test
    
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Entrenar el modelo
    history = model.fit(
        x_train_resized, y_train,
        validation_data=(x_test_resized, y_test),
        epochs=15,
        batch_size=64
    )
    
    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(x_test_resized, y_test)
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")
    
    # Guardar resultados
    results[model_name] = {
        "model": model,
        "history": history,
        "test_accuracy": test_accuracy
    }
    
    # Guardar el modelo
    model.save(f"cifar10_{model_name}.h5")

# 4. Mostrar los resultados
for model_name, data in results.items():
    print(f"{model_name} - Test Accuracy: {data['test_accuracy']:.4f}")

# 5. Graficar métricas del mejor modelo
best_model_name = max(results, key=lambda x: results[x]["test_accuracy"])
history = results[best_model_name]["history"]

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'{best_model_name} Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title(f'{best_model_name} Loss')
plt.show()