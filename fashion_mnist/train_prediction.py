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

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()

    df = pd.DataFrame(columns=['image', 'label'])
    for i, image in enumerate(train_images):
        df.loc[i] = ({'image': image, 'label': train_labels[i]})

    with open(output_dataset_train_data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


train_data_load_op = components.create_component_from_func(
    train_data_load, base_image='tensorflow/tensorflow',
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
)
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def train_op(
        train_dataset: InputPath('Dataset'),
        pre_model: InputPath('TFModel'),
        trained_model : OutputPath('TFModel')
) :
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
    predict_task = model_prediction_op(
        preprocess_task.outputs['data'],
        train_task.outputs['trained_model']
    )
    print_task1 = print_text(predict_task.outputs['predict'], predict_task.outputs['label'])


if __name__ == '__main__':
    import kfp

    kfp.compiler.Compiler().compile(fashion_mnist_pipeline, 'fashion_mnist_pipeline.yaml')