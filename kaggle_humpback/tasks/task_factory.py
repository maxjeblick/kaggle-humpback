from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from kaggle_humpback.tasks.landmark_detector import LandmarkDetector
from kaggle_humpback.tasks.identifier import Identifier


def get_task(config):
    f = globals().get(config.task.name)
    return f(config)
