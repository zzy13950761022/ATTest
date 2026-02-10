import pytest
import tempfile
import tensorflow as tf
from tensorflow.python.framework import graph_io
from unittest import mock

# 创建一个简单的测试
def test_simple():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建一个简单的图
        graph = tf.Graph()
        with graph.as_default():
            a = tf.constant(1.0, name="a")
        
        # 尝试mock
        with mock.patch('tensorflow.python.framework.graph_io.file_io.atomic_write_string_to_file') as mock_write:
            mock_write.return_value = None
            with mock.patch('tensorflow.python.framework.graph_io.file_io.recursive_create_dir') as mock_create_dir:
                mock_create_dir.return_value = None
                
                result = graph_io.write_graph(
                    graph_or_graph_def=graph,
                    logdir=tmpdir,
                    name="test.pbtxt",
                    as_text=True
                )
                
                print(f"Result: {result}")
                print(f"Mock write called: {mock_write.called}")
                print(f"Mock create dir called: {mock_create_dir.called}")

if __name__ == "__main__":
    test_simple()