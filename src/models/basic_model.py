from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1.0/255, input_shape=input_shape),

            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(), 

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Compile with RMSprop optimizer and categorical crossentropy loss
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )