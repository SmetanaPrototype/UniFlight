"""
Юнит-тесты для Rocket_parser.

Запуск:
    pytest test_rocket_parser.py -v
"""
import math
import pytest
import numpy as np

from rocket_parser import Rocket_parser
from rocket_parser import United_data, Length_dataset, Coord_element
import rocket_parser_utils


# =====================================================================
# Фикстуры
# =====================================================================
@pytest.fixture(scope="module")
def soyuz():
    """Парсер Союза-2.1в — тандемная схема, 2 ступени."""
    return Rocket_parser("soyuz21v")


@pytest.fixture(scope="module")
def amur():
    """Парсер Амура — тандем, 2 ступени, reuse."""
    return Rocket_parser("amur")


ALL_ROCKETS = ["soyuz21v", "amur"]


# =====================================================================
# 1. Контейнеры
# =====================================================================
class TestContainers:
    def test_coord_element_end(self):
        c = Coord_element(2.0, 3.0)
        assert c.start == 2.0
        assert c.length == 3.0
        assert c.end == 5.0

    def test_united_data_block_count(self):
        u = United_data(block_num=3)
        assert len(u.blocks) == 4   # 3 ступени + Payload

    def test_united_data_wrong_length_raises(self):
        u = United_data(block_num=2)
        with pytest.raises(ValueError, match="не совпадает"):
            u.set_length_dataset([Length_dataset()])

