from kfp import components, dsl
from kfp.components import InputPath, OutputPath
import kfp

def train_data_load(
        output_dataset_train_data: OutputPath('Dataset')
):
    import tensorflow as tf
    import pandas as pd
    import pickle
    import tensorflow_datasets as tfds

    ds, ds_info = tfds.load('beans', split='train', shuffle_files=True, with_info=True)

    df = pd.DataFrame(columns=['image', 'label'])
    for i, image in enumerate(ds):
        df.loc[i] = ({'image': image['image'], 'label': image['label']})

    with open(output_dataset_train_data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


train_data_load_op = components.create_component_from_func(
    train_data_load, base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2', 'tensorflow-datasets']
)

def flip_left_right(
        pre_data:InputPath('Dataset'),
        data: OutputPath('Dataset')
):
    import tensorflow as tf
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

    new_image = []
    new_label = []
    for i, image in enumerate(images):
        result = tf.image.flip_left_right(image)
        new_image.append(result)
        new_label.append(labels[i])

    df = pd.DataFrame(columns=['image', 'label'])

    for i, image in enumerate(new_image):
        df.loc[i] = ({'image': image, 'label': new_label[i]})
    with open(data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


flip_left_right_op = components.create_component_from_func(
    flip_left_right, base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2', 'numpy'])

##########################################################################
def rotation(
        pre_data:InputPath('Dataset'),
        data: OutputPath('Dataset')
):
    import tensorflow as tf
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

    new_image = []
    new_label = []
    for i, image in enumerate(images):
        result = tf.image.rot90(image)
        new_image.append(result)
        new_label.append(labels[i])


    df = pd.DataFrame(columns=['image', 'label'])

    for i, image in enumerate(new_image):
        df.loc[i] = ({'image': image, 'label': new_label[i]})
    with open(data, 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


rotation_op = components.create_component_from_func(
    rotation, base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2', 'numpy'])



def merge_data_generation(
        origin_data:InputPath('Dataset'),
        re_data:InputPath('Dataset'),
        ho_data:InputPath('Dataset'),
        data: OutputPath('Dataset')
):
    import numpy as np
    import pickle
    import pandas as pd

    with open(origin_data, 'rb') as file:
        data_1 = pickle.load(file)
    with open(re_data, 'rb') as file:
        data_2 = pickle.load(file)
    with open(ho_data, 'rb') as file:
        data_3 = pickle.load(file)

    result_temp = pd.concat([data_1,data_2], ignore_index=True)
    result = pd.concat([result_temp,data_3], ignore_index=True)

    with open(data, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

merge_data_generation_op = components.create_component_from_func(
    merge_data_generation, base_image='tensorflow/tensorflow',
    packages_to_install=['pandas==1.4.2', 'numpy'])

@dsl.pipeline(name='tensorflow beans dataset augmentation pipeline example')
def beans_data_augmentation_pipeline():
    train_data_load_task = train_data_load_op()
    resize_and_rescale_task = flip_left_right_op(
        train_data_load_task.outputs['output_dataset_train_data']
    )

    horizontal_and_vertical_task = rotation_op(
        train_data_load_task.outputs['output_dataset_train_data']
    )
    merge_data_generation_task = merge_data_generation_op(
        train_data_load_task.outputs['output_dataset_train_data'],
        resize_and_rescale_task.outputs['data'],
        horizontal_and_vertical_task.outputs['data']
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(beans_data_augmentation_pipeline, 'beans_condition_example.yaml')