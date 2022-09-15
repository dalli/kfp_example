from typing import NamedTuple

import kfp
from kfp.components import func_to_container_op
import requests

USERNAME = 'xxx@xxx.xxx'
PASSWORD = 'xxxxxx'
NAMESPACE = 'xxxxxx'
HOST = 'http://xxx.xxx.xxx.xxx:xxxx'

session = requests.Session()
response = session.get(HOST)

login_data = {"login": USERNAME, "password": PASSWORD}
session.post(response.url, data=login_data)
session_cookie = session.cookies.get_dict()["authservice_session"]

client = kfp.Client(
    host=f"{HOST}/pipeline",
    namespace=f"{NAMESPACE}",
    cookies=f"authservice_session={session_cookie}",
)
print(client.list_pipelines())

@func_to_container_op
def print_small_text(text: str):
    print(text)

def connect_example_pipeline(text1: str, text2: str):
    print_small_text(text1)
    print_small_text(text2)

client.create_run_from_pipeline_func(connect_example_pipeline, arguments={'text1' : 'hello world', 'text2' : 'test2'})