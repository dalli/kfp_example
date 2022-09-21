import kfp
from kfp import components, dsl
from kfp.components import func_to_container_op

@func_to_container_op
def node_A() -> str:
    task_A = 'A'
    print(task_A)
    return task_A

@func_to_container_op
def node_B(B: str) -> str:
    task_B = f'{B} -> B'
    print(task_B)
    return task_B

@dsl.pipeline(name='pipeline example')
def connect_example_pipeline():
    node_A_task = node_A()
    node_B_task = node_B(node_A_task.output)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(connect_example_pipeline, 'node2.yaml')