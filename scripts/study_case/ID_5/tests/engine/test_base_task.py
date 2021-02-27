import pytest
from scripts.study_case.MatchZoo_py.matchzoo.engine.base_task import BaseTask


def test_base_task_instantiation():
    with pytest.raises(TypeError):
        BaseTask()
