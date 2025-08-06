from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam

class HyperModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            layers.RandomFlip("horizontal", input_shape=input_shape),
            layers.RandomRotation(0.1),

            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Dropout(0.25),

            layers.Flatten(),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(categories_count, activation='softmax')
        ])

    def _compile_model(self):
        self.model.compile(
          optimizer=Adam(learning_rate=1e-4),
          loss='categorical_crossentropy',
          metrics=['accuracy']
        )
