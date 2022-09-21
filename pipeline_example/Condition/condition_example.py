import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from typing import NamedTuple
@func_to_container_op
def node_A() -> NamedTuple('Outputs', [('workflow', str), ('number', list)]):
    task_A = 'A'
    import random
    number1 = random.randint(1,10)
    number2 = random.randint(1,10)

    return (task_A, [number1, number2])

@func_to_container_op
def node_B() -> NamedTuple('Outputs', [('workflow', str), ('cal', str)]) :
    task_B = 'B'
    import random
    calculator_list = ['plus', 'minus', 'multiply', 'division']
    cal = calculator_list[random.randint(0,3)]
    return (task_B, cal)

@func_to_container_op
def node_C_plus(number : list, C1:str, C2:str) -> NamedTuple('Outputs', [('workflow', str), ('result', float)]) :
    task_C = f'{(C1, C2)} -> C_plus'
    print(task_C)
    result = sum(number)
    # result = int(number[0]) + int(number[1])
    return (task_C, result)

@func_to_container_op
def node_D_minus(number : list, D1:str, D2:str) -> NamedTuple('Outputs', [('workflow', str), ('result', float)]) :
    task_D = f'{(D1, D2)} -> D_minus'
    print(task_D)
    result = int(number[0]) - int(number[1])
    return (task_D, result)

@func_to_container_op
def node_E_multiply(number : list, E1:str, E2:str) -> NamedTuple('Outputs', [('workflow', str), ('result', float)]):
    task_E = f'{(E1, E2)} -> E_multiply'
    print(task_E)
    result = int(number[0]) * int(number[1])
    return (task_E, result)

@func_to_container_op
def node_F_division(number : list, F1:str, F2:str) -> NamedTuple('Outputs', [('workflow', str), ('result', float)]):
    task_F = f'{(F1, F2)} -> F_division'
    print(task_F)
    result = int(number[0]) / int(number[1])
    return (task_F, result)


def connect_example_pipeline():
    node_A_task = node_A()
    node_B_task = node_B()

    with dsl.Condition(node_B_task.outputs['cal'] == 'plus'):
        node_C_plus_task = node_C_plus(node_A_task.outputs['number'], node_A_task.outputs['workflow'], node_B_task.outputs['workflow'])
    with dsl.Condition(node_B_task.outputs['cal'] == 'minus'):
        node_D_minus_task = node_D_minus(node_A_task.outputs['number'], node_A_task.outputs['workflow'], node_B_task.outputs['workflow'])
    with dsl.Condition(node_B_task.outputs['cal'] == 'multiply'):
        node_E_multiply_task = node_E_multiply(node_A_task.outputs['number'], node_A_task.outputs['workflow'], node_B_task.outputs['workflow'])
    with dsl.Condition(node_B_task.outputs['cal'] == 'division'):
        node_F_division_task = node_F_division(node_A_task.outputs['number'], node_A_task.outputs['workflow'], node_B_task.outputs['workflow'])

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(connect_example_pipeline, 'condition_example.yaml')