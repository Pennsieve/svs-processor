#!/usr/bin/env python
import os

import pytest
from base_processor.tests import init_ssm, setup_processor
from moto import mock_dynamodb2
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from svs_processor import SVSProcessor

test_processor_data = [
    'small_region.svs'
]


@pytest.fixture(
    scope='function'
)
def task(request):
    yield tsk

    tsk._cleanup()

    mock_dynamodb2().stop()


@pytest.mark.parametrize("filename", test_processor_data)
def test_svs_processor(filename):
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    inputs = {'file': os.path.join('/test-resources', filename), 'image_quality': 0.8}
    task = SVSProcessor(inputs=inputs)

    setup_processor(task)

    # run
    task.run()

    assert os.path.isfile('%s-zoomed/dimensions.json' % os.path.basename(filename))
    assert os.path.isfile('%s-zoomed/dim_Z_slice_1_dim_T_slice_0.dzi' % os.path.basename(filename))
    assert os.path.isfile('metadata.json')
