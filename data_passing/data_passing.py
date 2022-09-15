from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.components import func_to_container_op


@func_to_container_op
def print_small_text(text: str):
    '''Print small text'''
    print(text)

@func_to_container_op
def print_small_int(text: int):
    '''Print small text'''
    print(text)

@func_to_container_op
def print_two_arguments(text: str, number: int):
    print('Text={}'.format(text))
    print('Number={}'.format(str(number)))

@func_to_container_op
def one_small_output_str() -> str:
    return 'one_output'

@func_to_container_op
def one_small_output_int() -> int:
    return 111

@func_to_container_op
def two_small_outputs() -> NamedTuple('Outputs', [('text', str), ('number', int)]):
    return ("two_output", 222)

@dsl.pipeline(name='data passing')
def data_passing_pipeline():
    '''Pipeline that passes small constant string to to consumer'''
    single_small_output1_task = one_small_output_str()
    single_small_output2_task = one_small_output_int()

    tuple_small_output_task = two_small_outputs()

    print_one1_task = print_small_text(single_small_output1_task.output)
    print_one2_task = print_small_int(single_small_output2_task.output)

    print_two1_task = print_two_arguments(tuple_small_output_task.outputs['text'], tuple_small_output_task.outputs['number'])
    print_two1_task = print_two_arguments(single_small_output1_task.output, single_small_output2_task.output)

if __name__ == '__main__':
    # Compiling the pipeline
    # yaml or tar.gz 로 저장 가능
    kfp.compiler.Compiler().compile(data_passing_pipeline, 'data_passing.tar.gz')
