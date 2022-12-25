import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, ReLU, BatchNormalization, Dropout, DepthwiseConv2D, Add, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RandomContrast
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

class Train():

    def __init__(self, path : str, checkpoint_path= 'tmp/hichi'):
        self.path = path
        sys.path.append(path)
        Train.dataset(self)
        Train.train(self,checkpoint_filepath=checkpoint_path)

    def dataset(self):
        self.calass_names = ['alphabet', 'number']
        self.batch_size = 128
        IMG_SIZE = (32, 32)
        training_directory = f"{self.path}/train"
        valid_directory = f"{self.path}/valid"

        self.train_dataset = image_dataset_from_directory(training_directory,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
        self.validation_dataset = image_dataset_from_directory(valid_directory,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
    
    def data_augmenter():

        data_augmentation = tf.keras.Sequential()
        data_augmentation.add(RandomContrast(0.2)) 

        return data_augmentation

    def block_c(x, power, last_filters):

        batch, input_height, input_width, input_channels = x.shape
        x = Conv2D(power*input_channels, (1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = DepthwiseConv2D((3, 3), strides=(2, 2),padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = Conv2D(last_filters, (1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        
        return x
        
    def block_b(x, power, last_filters):
        
        batch, input_height, input_width, input_channels = x.shape
        x = Conv2D(power*input_channels, (1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = DepthwiseConv2D((3, 3), padding='same')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = Conv2D(last_filters, (1, 1), padding='same')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        
        return x

    def block_a(x, power, last_filters):
        
        batch, input_height, input_width, input_channels = x.shape
        x = BatchNormalization(axis=3)(x, training=True)
        input_x = x
        x = Conv2D(power*input_channels, (1, 1), padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = DepthwiseConv2D((3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = Conv2D(last_filters, (1, 1), padding='same')(x)
        #x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = Add()([x, input_x])
        
        return x

    def conv_model(input_shape=(32, 32, 3)):

        x_input = Input(shape=input_shape)
        x = Train.data_augmenter()(x_input)
        x = Conv2D(8, (3, 3), strides=(2, 2),padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = Train.block_c(x, 1, 8)
        x = Train.block_c(x, 2, 16)
        x = Train.block_a(x, 2, 16)
        x = Train.block_c(x, 2, 24)
        x = Train.block_a(x, 2, 24)
        x = Train.block_b(x, 2, 32)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=x_input, outputs=x)
        
        return model

    def plot_history(self):
        
        acc = [0.] + self.history.history['accuracy']
        val_acc = [0.] + self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    def train(self, model=None, base_learning_rate = 0.01, initial_epochs = 25, checkpoint_filepath = 'tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(2)', monitor = 'val_accuracy'):
        
        self.model = model
        
        if model == None:
            model = Train.conv_model()
        
        model.compile(optimizer=Adam(lr=base_learning_rate), loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor=monitor,mode='max', save_best_only=True)
        self.history = model.fit(self.train_dataset, validation_data=self.validation_dataset, epochs=initial_epochs, callbacks=model_checkpoint_callback)
        print('Training process finished!')
        return self.history
        