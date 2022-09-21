import kfp
from kfp import components, dsl
from kfp.components import func_to_container_op

from typing import NamedTuple
@func_to_container_op
def node_A() -> NamedTuple('Outputs', [('first', str), ('second', str)]):
    task_A = 'A'
    print(task_A)
    return (task_A, task_A)

@func_to_container_op
def node_B(B: str) -> NamedTuple('Outputs', [('first', str), ('second', str)]):
    task_B = f'{B} -> B'
    print(task_B)
    return (task_B, task_B)

@func_to_container_op
def node_C(C: str) -> str:
    task_C = f'{C} -> C'
    print(task_C)
    return task_C

@func_to_container_op
def node_D(D: str) -> str:
    task_D = f'{D} -> D'
    print(task_D)
    return task_D

@func_to_container_op
def node_E(E: str) -> str:
    task_E = f'{E} -> E'
    print(task_E)
    return task_E

@func_to_container_op
def node_F(F: str) -> str:
    task_F = f'{F} -> F'
    print(task_F)
    return task_F

@dsl.pipeline(name='pipeline example')
def connect_example_pipeline():
    node_A_task = node_A()

    node_B_task = node_B(node_A_task.outputs['first'])

    node_D_task = node_D(node_B_task.outputs['first'])
    node_E_task = node_E(node_B_task.outputs['second'])

    node_C_task = node_C(node_A_task.outputs['second'])

    node_F_task = node_F(node_C_task.output)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(connect_example_pipeline, 'node_example.yaml')