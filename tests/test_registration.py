
import unittest

from energy_fault_detector.registration import Registry


class SomeClass:
    """A test class"""
    def __init__(self, x: int):
        self.x: int = x

    def add_one(self):
        self.x += 1


class AnotherClass:
    """A test class"""
    def __init__(self, x: int):
        self.x: int = x

    def minus_one(self):
        self.x -= 1


class TestRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.registry: Registry = Registry()
        # Register using module paths
        self.registry.register('tests.test_registration.SomeClass', 'class_type', ['SomeClass', 'some_class'])
        self.registry.register('tests.test_registration.AnotherClass', 'class_type', ['another_class', 'a_class'])
        self.registry.register('tests.test_registration.SomeClass', 'another_class_type', ['SomeClass'])

    def test_register(self):
        exp_dictinary = {
            'class_type': {
                'some_class': 'tests.test_registration.SomeClass',
                'SomeClass': 'tests.test_registration.SomeClass',
                'another_class': 'tests.test_registration.AnotherClass',
                'a_class': 'tests.test_registration.AnotherClass',
            },
            'another_class_type': {
                'SomeClass': 'tests.test_registration.SomeClass',
            }
        }
        self.assertDictEqual(exp_dictinary, self.registry.registry)

        self.registry.register('tests.test_registration.SomeClass', 'another_class_type', ['some_class'])
        exp_dictionary2 = {
            'class_type': {
                'some_class': 'tests.test_registration.SomeClass',
                'SomeClass': 'tests.test_registration.SomeClass',
                'another_class': 'tests.test_registration.AnotherClass',
                'a_class': 'tests.test_registration.AnotherClass',
            },
            'another_class_type': {
                'SomeClass': 'tests.test_registration.SomeClass',
                'some_class': 'tests.test_registration.SomeClass',
            }
        }
        self.assertDictEqual(exp_dictionary2, self.registry.registry)

    def test_get(self):
        some_class = self.registry.get('class_type', 'some_class')
        an_instance = some_class(1)
        self.assertEqual(an_instance.x, 1)

    def test__check_exists(self):
        with self.assertRaises(ValueError):
            self.registry.register('tests.test_registration.SomeClass', 'class_type', ['some_class'])
