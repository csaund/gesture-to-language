import unittest
from GestureClusterer import *

# 1. find and load a couple of canonical examples with keyframes
#     gesture_features = [
#         self._palm_vert(gesture, 'l'),
#         self._palm_horiz(gesture, 'l'),
#         self._palm_vert(gesture, 'r'),
#         self._palm_horiz(gesture, 'r'),
#         self._max_hands_apart(gesture),
#         self._min_hands_together(gesture),
#         #   x_oscillate,
#         #   y_oscillate,
#         self._wrists_up(gesture, 'r'),
#         self._wrists_up(gesture, 'l'),
#         self._wrists_down(gesture, 'r'),
#         self._wrists_down(gesture, 'l'),
#         self._wrists_outward(gesture),
#         self._wrists_inward(gesture),
#         self._max_wrist_velocity(gesture, 'r'),
#         self._max_wrist_velocity(gesture, 'l')
#         #   wrists_sweep,
#         #   wrist_arc,
#         #   r_hand_rotate,
#         #   l_hand_rotate,
#         #   hands_cycle
#     ]

class TestMotionFeatures(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()