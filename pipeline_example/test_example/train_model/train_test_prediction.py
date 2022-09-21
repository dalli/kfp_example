# Copyright 2021 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Two step v2-compatible pipeline."""
import random

from kfp import components, dsl
from kfp.components import InputPath, OutputPath
from kfp.components import func_to_container_op
from typing import NamedTuple


def train_data_load(
        output_dataset_train_data: OutputPath('Dataset')
):
    import tensorflow as tf
    import pandas as pd
    import pickle

    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (_, _) = mnist.load_data()

    df = pd.DataFrame(columns=['image', 'label'])
    for i, image in enumerate(train_images):
        df.loc[i] = ({'image': image, 'label': train_labels[i]})

    with open(output_dataset_train_data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


train_data_load_op = components.create_component_from_func(
    train_data_load, base_image='tensorflow/tensorflow',
    #output_component_file = 'train_data.pickle',
    packages_to_install=['pandas==1.4.2']
)
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
def test_data_load(
        output_dataset_test_data: OutputPath('Dataset')
):
    import tensorflow as tf
    import pandas as pd
    import pickle

    mnist = tf.keras.datasets.mnist
    (_, _), (test_images, test_labels) = mnist.load_data()

    df = pd.DataFrame(columns=['image', 'label'])
    for i, image in enumerate(test_images):
        df.loc[i] = ({'image': image, 'label': test_labels[i]})

    with open(output_dataset_test_data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


test_data_load_op = components.create_component_from_func(
    test_data_load, base_image='tensorflow/tensorflow',
    #output_component_file = 'train_data.pickle',
    packages_to_install=['pandas==1.4.2']
)

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
def preprocess(
        pre_data:InputPath('Dataset'),
        data: OutputPath('Dataset')
):
    import numpy as np
    import pickle
    import pandas as pd

    images = []
    labels = []
    with open(pre_data, 'rb') as file:
        tr_data = pickle.load(file)

    for i, item in enumerate(tr_data['image']):
        images.append(item)
        labels.append(tr_data['label'][i])
    images = np.array(images)
    labels = np.array(labels)

    images = images/255.0

    df = pd.DataFrame(columns=['image', 'label'])
    for i, image in enumerate(images):
        df.loc[i] = ({'image': image, 'label': labels[i]})

    with open(data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


preprocess_op = components.create_component_from_func(
    preprocess, base_image='python:3.9',
    packages_to_install=['numpy==1.23.2', 'pandas==1.4.2']
)
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
def model_generation(pretrain_model : OutputPath('TFModel')) :
    import tensorflow as tf
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    keras_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    keras_model.save(pretrain_model)

load_model_op = components.create_component_from_func(
    model_generation, base_image='tensorflow/tensorflow'
    #output_component_file = 'model.h5'
)
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def train_op(
        train_dataset: InputPath('Dataset'),
        pre_model: InputPath('TFModel'),
        trained_model : OutputPath('TFModel')
) :
    '''Dummy Training Step.'''
    import pickle
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import pandas as pd

    with open(train_dataset, 'rb') as file:
        tr_data = pickle.load(file)

    images = []
    labels = []
    for i, item in enumerate(tr_data['image']) :
        images.append(item)
        labels.append(tr_data['label'][i])
    images = np.array(images)
    labels = np.array(labels)

    model = keras.models.load_model(pre_model)

    model.fit(images, labels, epochs=20)

    model.save(trained_model)

train_result_op = components.create_component_from_func(
    train_op,
    base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2']
)
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
def model_test(
    test_dataset: InputPath('Dataset'),
    trained_model : InputPath('TFModel'),
) -> NamedTuple('Outputs', [('accuracy', str), ('loss', str)]):
    from tensorflow import keras
    import pickle
    import pandas as pd
    import numpy as np

    with open(test_dataset, 'rb') as file:
        tr_data = pickle.load(file)

    images = []
    labels = []
    for i, item in enumerate(tr_data['image']) :
        images.append(item)
        labels.append(tr_data['label'][i])
    images = np.array(images)
    labels = np.array(labels)

    model = keras.models.load_model(trained_model)
    model.summary()

    loss, acc = model.evaluate(images, labels, verbose=2)

    acc_text = f'trained model test accuracy : {acc*100} %'
    loss_text = f'trained model test loss : {loss}'

    return (acc_text, loss_text)

model_test_op = components.create_component_from_func(
    model_test,
    base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2']
)

def model_prediction(
    test_dataset: InputPath('Dataset'),
    trained_model : InputPath('TFModel')
) -> NamedTuple('Outputs', [('predict', str), ('label', str)]):
    from tensorflow import keras
    import tensorflow as tf
    import pickle
    import pandas as pd
    import numpy as np
    import random

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with open(test_dataset, 'rb') as file:
        tr_data = pickle.load(file)

    images = []
    labels = []
    for i, item in enumerate(tr_data['image']):
        images.append(item)
        labels.append(tr_data['label'][i])
    images = np.array(images)
    labels = np.array(labels)

    test_num = random.randrange(1,1000)

    model = keras.models.load_model(trained_model)

    predic_image = images[test_num]
    predic_label = labels[test_num]

    test = tf.expand_dims(predic_image, 0)
    predictions_single = model.predict(test)
    predict_value = tf.math.argmax(tf.nn.softmax(predictions_single[0]))

    predict_value = f'predict result : {class_names[predict_value]}'
    label_value = f'label result: {class_names[predic_label]}'

    return (predict_value, label_value)

model_prediction_op = components.create_component_from_func(
    model_prediction,
    base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2']
)
@func_to_container_op
def print_text(text1: str, text2: str):
    print(text1)
    print(text2)

@dsl.pipeline(name='tak test fashion mnist pipeline')
def fashion_mnist_pipeline():
    train_data_load_task = train_data_load_op()
    preprocess_task = preprocess_op(
        train_data_load_task.outputs['output_dataset_train_data']
    )
    model_task = load_model_op()
    train_task = train_result_op(
        preprocess_task.outputs['data'],
        model_task.outputs['pretrain_model']
    )

    test_data_load_task = test_data_load_op()
    preprocess_task = preprocess_op(
        test_data_load_task.outputs['output_dataset_test_data']
    )
    test_task = model_test_op(
        preprocess_task.outputs['data'],
        train_task.outputs['trained_model']
    )
    predict_task = model_prediction_op(
        preprocess_task.outputs['data'],
        train_task.outputs['trained_model']
    )
    print_task1 = print_text(test_task.outputs['accuracy'], test_task.outputs['loss'])
    print_task2 = print_text(predict_task.outputs['predict'], predict_task.outputs['label'])

if __name__ == '__main__':
    # Compiling the pipeline
    import kfp
    kfp.compiler.Compiler().compile(fashion_mnist_pipeline, __file__ + '.yaml')