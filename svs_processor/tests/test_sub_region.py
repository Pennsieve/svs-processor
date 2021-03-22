#!/usr/bin/env python
import os

import pytest
from moto import mock_dynamodb2
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from svs_processor import SVSProcessor

from base_processor.tests import init_ssm, setup_processor

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
def test_svs_processor_xyzct(filename):
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for c in range(2):
                    for t in range(2):
                        # Create sub_region file
                        open('sub_x_{}_2_y_{}_2_z_{}_2_c_{}_2_t_{}_2.txt'.format(x, y, z, c, t), 'w').write("")
                        # Setup task with inputs
                        inputs = {
                            'file': os.path.join('/test-resources', filename),
                            'sub_region_file': 'sub_x_{}_2_y_{}_2_z_{}_2_c_{}_2_t_{}_2.txt'.format(x, y, z, c, t)
                        }
                        task = SVSProcessor(inputs=inputs)
                        setup_processor(task)
                        # run
                        task.run()

@pytest.mark.parametrize("filename", test_processor_data)
def test_svs_processor_xy(filename):
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    for x in range(2):
        for y in range(2):
            for z in range(1):
                for c in range(1):
                    for t in range(1):
                        # Create sub_region file
                        open('sub_x_{}_2_y_{}_2_z_{}_2_c_{}_2_t_{}_2.txt'.format(x, y, z, c, t), 'w').write("")
                        # Setup task with inputs
                        inputs = {
                            'file': os.path.join('/test-resources', filename),
                            'sub_region_file': 'sub_x_{}_2_y_{}_2_z_{}_2_c_{}_2_t_{}_2.txt'.format(x, y, z, c, t)
                        }
                        task = SVSProcessor(inputs=inputs)
                        setup_processor(task)
                        # run
                        task.run()