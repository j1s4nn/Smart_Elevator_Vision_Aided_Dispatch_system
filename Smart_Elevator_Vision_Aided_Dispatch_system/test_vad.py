"""
=============================================================================
VAD System – Unit Tests
=============================================================================
Run with:  python test_vad.py
No camera or GPU needed.  All tests use synthetic data.
=============================================================================
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from vad_main import (
    CentroidTracker,
    FloorCall,
    ElevatorState,
    GhostCallFilter,
    CapacityManagementModule,
    SmartDispatchEngine,
)


# =============================================================================
class TestCentroidTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=50)

    def test_registers_new_objects(self):
        boxes = [(10, 10, 50, 90), (100, 10, 140, 90)]
        result = self.tracker.update(boxes)
        self.assertEqual(len(result), 2, "Should register 2 unique IDs")

    def test_maintains_id_across_frames(self):
        boxes = [(10, 10, 50, 90)]
        self.tracker.update(boxes)
        first_id = list(self.tracker.objects.keys())[0]

        # Move the box slightly (within max_distance)
        boxes2 = [(12, 12, 52, 92)]
        self.tracker.update(boxes2)
        second_id = list(self.tracker.objects.keys())[0]

        self.assertEqual(first_id, second_id, "ID must persist across frames")

    def test_deregisters_after_disappearance(self):
        self.tracker.update([(10, 10, 50, 90)])
        # Send 6 empty frames (threshold = 5)
        for _ in range(6):
            self.tracker.update([])
        self.assertEqual(len(self.tracker.objects), 0, "Object should deregister")

    def test_count_property(self):
        self.tracker.update([(0, 0, 20, 40), (50, 0, 70, 40), (100, 0, 120, 40)])
        self.assertEqual(self.tracker.count, 3)


# =============================================================================
class TestGhostCallFilter(unittest.TestCase):

    def setUp(self):
        self.gcf = GhostCallFilter(ghost_threshold_frames=3)

    def test_valid_call_with_people(self):
        call = FloorCall(floor_id=2, direction="UP")
        result = self.gcf.update(2, call, people_count=2)
        self.assertTrue(result, "Call with people should remain valid")

    def test_ghost_call_cancellation(self):
        call = FloorCall(floor_id=3, direction="DOWN")
        # 4 frames of nobody (threshold = 3)
        for i in range(4):
            self.gcf.update(3, call, people_count=0)
        self.assertFalse(call.is_active, "Call should be cancelled as ghost")

    def test_counter_resets_when_person_detected(self):
        call = FloorCall(floor_id=4, direction="UP")
        self.gcf.update(4, call, people_count=0)
        self.gcf.update(4, call, people_count=0)
        self.gcf.update(4, call, people_count=1)  # Person detected → reset
        self.gcf.update(4, call, people_count=0)
        # Only 1 empty frame after reset → should NOT cancel
        self.assertTrue(call.is_active, "Call should still be valid after reset")

    def test_already_cancelled_call(self):
        call = FloorCall(floor_id=5, direction="UP")
        call.is_active = False
        result = self.gcf.update(5, call, people_count=5)
        self.assertFalse(result, "Already-cancelled call must stay cancelled")


# =============================================================================
class TestCapacityManagementModule(unittest.TestCase):

    def setUp(self):
        self.cmm = CapacityManagementModule()

    def test_allows_pickup_when_not_full(self):
        elev = ElevatorState(elevator_id=0, occupancy=5, max_capacity=12)
        result = self.cmm.check(elev, requesting_floor=3)
        self.assertTrue(result)

    def test_blocks_pickup_when_full(self):
        elev = ElevatorState(elevator_id=0, occupancy=12, max_capacity=12)
        result = self.cmm.check(elev, requesting_floor=7)
        self.assertFalse(result, "Full elevator must block pickups")

    def test_blocks_when_overcapacity(self):
        elev = ElevatorState(elevator_id=0, occupancy=13, max_capacity=12)
        self.assertFalse(self.cmm.check(elev, 2))


# =============================================================================
class TestSmartDispatchEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SmartDispatchEngine(w1=10.0, w2=1.0, w3=0.1)

    def _make_elevator(self, floor=0, occupancy=0, direction="IDLE"):
        e = ElevatorState(elevator_id=0, current_floor=floor,
                          direction=direction, occupancy=occupancy, max_capacity=12)
        return e

    def test_higher_people_count_wins(self):
        elev = self._make_elevator(floor=0)
        call_small = FloorCall(floor_id=3, direction="UP", people_count=1)
        call_large = FloorCall(floor_id=3, direction="UP", people_count=5)
        time.sleep(0.01)  # ensure same age

        score_small = self.engine.priority_score(call_small, elev)
        score_large = self.engine.priority_score(call_large, elev)
        self.assertGreater(score_large, score_small,
                           "Larger crowd should produce a higher P_i")

    def test_closer_floor_wins_if_same_crowd(self):
        elev = self._make_elevator(floor=5)
        call_near = FloorCall(floor_id=6, direction="UP", people_count=2)
        call_far  = FloorCall(floor_id=10, direction="UP", people_count=2)
        time.sleep(0.01)

        score_near = self.engine.priority_score(call_near, elev)
        score_far  = self.engine.priority_score(call_far,  elev)
        self.assertGreater(score_near, score_far,
                           "Closer floor should score higher when crowd is equal")

    def test_ghost_call_scores_zero(self):
        elev = self._make_elevator()
        call = FloorCall(floor_id=4, direction="UP", people_count=0)
        call.is_active = False  # Already cancelled
        score = self.engine.priority_score(call, elev)
        self.assertEqual(score, 0.0)

    def test_full_elevator_scores_zero(self):
        elev = self._make_elevator(occupancy=12)
        call = FloorCall(floor_id=3, direction="DOWN", people_count=3)
        score = self.engine.priority_score(call, elev)
        self.assertEqual(score, 0.0, "Full elevator should produce score=0")

    def test_select_best_stop_returns_highest(self):
        elev = self._make_elevator(floor=0)
        calls = [
            FloorCall(floor_id=2, direction="UP", people_count=1),
            FloorCall(floor_id=5, direction="UP", people_count=4),
            FloorCall(floor_id=8, direction="UP", people_count=2),
        ]
        best = self.engine.select_best_stop(calls, elev)
        self.assertEqual(best.floor_id, 5,
                         "Floor 5 with 4 people should be selected")

    def test_old_call_eventually_wins(self):
        """Anti-starvation: an old call with 1 person should beat a new call
           with 1 person at the same distance, given enough age."""
        elev = self._make_elevator(floor=0)
        old_call = FloorCall(floor_id=3, direction="UP", people_count=1)
        old_call.timestamp = time.time() - 120  # 2 minutes old

        new_call = FloorCall(floor_id=3, direction="UP", people_count=1)

        score_old = self.engine.priority_score(old_call, elev)
        score_new = self.engine.priority_score(new_call, elev)
        self.assertGreater(score_old, score_new,
                           "Older call should score higher due to W3 * A_i")


# =============================================================================
if __name__ == "__main__":
    print("Running VAD unit tests...\n")
    unittest.main(verbosity=2)
