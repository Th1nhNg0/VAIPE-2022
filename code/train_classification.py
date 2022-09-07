import datetime

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from tqdm import tqdm

image_width = 224
image_height = 224
batch_size = 64
split_size = 0.05
model_name = 'resnet50-bb-trainall' \
    + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

data_augmentation =tf.keras.Sequential([
                                tf.keras.layers.RandomZoom(0.2),
                                tf.keras.layers.RandomContrast(0.2),
                               tf.keras.layers.RandomRotation(1)])


def create_left():
    model = tf.keras.Sequential([Input(shape=(108, ))])
    return model


def create_right(width, height):
    input_shape = (height, width, 3)
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    base_model = \
        tf.keras.applications.resnet50.ResNet50(include_top=False,
            input_shape=input_shape, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=200, activation='relu')(x)
    model = Model(inputs, outputs)
    return (model, base_model)


def create_model():
    left_input = create_left()
    (right_input, right_base_model) = create_right(image_width,
            image_height)
    combinedInput = concatenate([left_input.output, right_input.output])
    x = Dense(200, activation='relu')(combinedInput)
    x = Dropout(0.2)(x)
    x = Dense(108, activation='softmax')(x)
    model = Model(inputs=[left_input.input, right_input.input],
                  outputs=x)
    return (model, right_base_model)


def image_d(pill_path):
    img = tf.io.read_file(pill_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_width, image_height])
    img = tf.cast(img, tf.float32)
    return img


def label_d(label):
    return tf.one_hot(label, 108)


def vector_d(vector):
    vector = tf.strings.substr(vector, 1, tf.strings.length(vector) - 2)
    vector = tf.strings.split(vector, sep=', ')
    vector = tf.strings.to_number(vector)

    return vector


if __name__ == '__main__':
    data = pd.read_csv('gen_train_data/combine_train.csv', index_col=0)
    data = data.sample(frac=1).reset_index(drop=True)
    (train_data, val_data) = train_test_split(data,
            test_size=split_size)

    vector = tf.data.Dataset.from_tensor_slices(train_data['vector'
            ]).map(vector_d)
    image = tf.data.Dataset.from_tensor_slices(train_data['pill_path'
            ]).map(image_d)
    label = tf.data.Dataset.from_tensor_slices(train_data['label'
            ]).map(label_d)
    train_dataset = tf.data.Dataset.zip(((vector, image),
            label)).shuffle(5000).batch(batch_size)

    vector = tf.data.Dataset.from_tensor_slices(val_data['vector'
            ]).map(vector_d)
    image = tf.data.Dataset.from_tensor_slices(val_data['pill_path'
            ]).map(image_d)
    label = tf.data.Dataset.from_tensor_slices(val_data['label'
            ]).map(label_d)
    val_dataset = tf.data.Dataset.zip(((vector, image),
            label)).batch(batch_size)

    (model, right_base_model) = create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # fine tune
    right_base_model.trainable = True
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])  # Low learning rate
    model.fit(train_dataset, validation_data=val_dataset, epochs=90)
    model.save(f'/app/models/{model_name}.h5')
    print(f'save model: /app/models/{model_name}.h5',)
