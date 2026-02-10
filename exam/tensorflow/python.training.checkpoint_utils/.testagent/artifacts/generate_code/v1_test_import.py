#!/usr/bin/env python3
"""Test import of checkpoint_utils."""

import numpy as np
import pytest
import tempfile
import os
import time
from unittest import mock

import tensorflow as tf
from tensorflow.python.training import checkpoint_utils

# Test that we can import the module
print("Successfully imported checkpoint_utils")

# Test that the module has the expected functions
assert hasattr(checkpoint_utils, 'load_checkpoint')
assert hasattr(checkpoint_utils, 'load_variable')
assert hasattr(checkpoint_utils, 'list_variables')
assert hasattr(checkpoint_utils, 'checkpoints_iterator')
assert hasattr(checkpoint_utils, 'init_from_checkpoint')

print("All expected functions are present")

# Test that checkpoint_management is accessible
assert hasattr(checkpoint_utils, 'checkpoint_management')
print("checkpoint_management is accessible")

# Test that py_checkpoint_reader is accessible
assert hasattr(checkpoint_utils, 'py_checkpoint_reader')
print("py_checkpoint_reader is accessible")

print("All tests passed!")