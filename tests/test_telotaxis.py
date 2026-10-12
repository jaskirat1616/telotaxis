"""Behaviour of the vision math, the onset timer, and the loaded pathway."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from telotaxis.decoder import Decoder
from telotaxis.vision import SceneObject, encode


CENTERS = np.linspace(-135, 135, 48)


class VisionTests(unittest.TestCase):
    def test_left_cue_drives_left_columns(self):
        drive, _ = encode([SceneObject("ball", "food", 1.0, 1.2, 0.4)], 0.0, 0.0, 0.0, CENTERS, stimulus=True)
        self.assertGreater(drive[CENTERS > 20].mean(), drive[CENTERS < -20].mean())

    def test_absent_is_zero(self):
        drive, _ = encode([SceneObject("ball", "food", 1.0, 0.0, 0.4)], 0.0, 0.0, 0.0, CENTERS, stimulus=False)
        self.assertEqual(float(drive.sum()), 0.0)

    def test_max_not_sum(self):
        one, _ = encode([SceneObject("big", "food", 2.0, 0.0, 0.8)], 0.0, 0.0, 0.0, CENTERS, stimulus=True)
        many = [SceneObject(f"s{i}", "food", 2.0, 0.02 * i, 0.05) for i in range(12)]
        piled, _ = encode(many, 0.0, 0.0, 0.0, CENTERS, stimulus=True)
        self.assertGreater(float(one.max()), float(piled.max()))


class DecoderTests(unittest.TestCase):
    def test_onset_then_no_stop(self):
        decoder = Decoder()
        speed, _ = decoder.update(40, 40, 0.015, 1.0)
        self.assertEqual(speed, 0.0)
        for _ in range(12):
            speed, _ = decoder.update(40, 40, 0.015, 1.6)
        self.assertGreater(speed, 0.2)
        speed, yaw = decoder.update(10, 40, 0.015, 3.0)
        self.assertGreater(speed, 0.2)
        self.assertLess(yaw, 0.0)


class WorldTests(unittest.TestCase):
    def test_forward_and_no_pair_contact(self):
        from telotaxis.world import World

        world = World(2, "car", seed=2)
        start = world.poses().copy()
        for _ in range(800):
            world.set_drive(np.array([0.4, 0.4]), np.zeros(2))
            world.step(1)
        for index in range(2):
            heading = start[index, 2]
            delta = world.poses()[index, :2] - start[index, :2]
            along = delta[0] * math.cos(heading) + delta[1] * math.sin(heading)
            self.assertGreater(along, 0.2)
        self.assertEqual(world.agent_agent_contacts(), 0)


if __name__ == "__main__":
    unittest.main()
