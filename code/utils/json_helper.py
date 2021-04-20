''''
 * @Descripttion: some json code
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-16 12:38:54
 '''
import json

from refile import smart_open


def load_json_items(json_path):
    """
    load json items from file path
    注：该文件的每一行是一个json格式的数据
    """
    with open(json_path, "r") as f:
        json_items = [json.loads(s) for s in f.readlines()]
    return json_items


def save_json_items(json_path, json_items):
    """
    save json items to file path
    注：该文件的每一行是一个json格式的数据
    """
    with open(json_path, "w") as f:
        f.write("\n".join([json.dumps(x) for x in json_items]))
