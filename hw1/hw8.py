"""
Решите задачу классификации изображений на наборе данных
с фотографиями животных (например, набор данных CIFAR-10).
"""


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import numpy as np


# Загружаем весь набор CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Классы животных в CIFAR-10
animal_classes = [2, 3, 4, 5, 6, 7]  # Птицы, кошки, олени, собаки, лягушки, лошади

# x = np.isin(y_train, animal_classes).flatten()
# print(y_train.shape)    # (50000, 1)
# print(X_train.shape)    # (50000, 32, 32, 3)
# print(x, x.shape)    # [ True False False ... False False False] (50000,)

# Фильтрация данных
train_mask = np.isin(y_train, animal_classes).flatten()
test_mask = np.isin(y_test, animal_classes).flatten()

X_train_animals, y_train_animals = X_train[train_mask], y_train[train_mask]
X_test_animals, y_test_animals = X_test[test_mask], y_test[test_mask]

# Перенумерация меток
class_mapping = {cls: i for i, cls in enumerate(animal_classes)}  # {2:0, 3:1, ..., 7:5}
y_train_animals, y_test_animals = y_train_animals.flatten(), y_test_animals.flatten()
y_train_animals = np.array([class_mapping[y] for y in y_train_animals])
y_test_animals = np.array([class_mapping[y] for y in y_test_animals])

# Нормализация
X_train_animals, X_test_animals = X_train_animals / 255.0, X_test_animals / 255.0

# include_top=False в ResNet50 - финальные полностью связанные (Dense) слои будут удалены.
# weights="imagenet" - модель уже обучена на большом наборе данных ImageNet и содержит предварительно настроенные веса.
# Загружаем предобученную модель ResNet50 без финального слоя
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(None, None, 3))

# Создаем кастомный классификатор
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Адаптирует модель к произвольному размеру входного изображения
    Dense(128, activation="relu"),
    Dense(len(animal_classes), activation="softmax")  # 6 классов животных
])

# Компиляция
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Вывод структуры модели
model.summary()

model.fit(X_train_animals, y_train_animals, epochs=10, validation_data=(X_test_animals, y_test_animals))

model.save("cifar10_animals_model.h5")
