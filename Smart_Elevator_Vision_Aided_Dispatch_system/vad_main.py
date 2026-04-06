"""
=============================================================================
Vision-Aided Dispatch (VAD) System - Complete Implementation
=============================================================================
Author  : Extended from Hossen Md Jisan's VAD Report
Version : 2.0 (Enhanced with PyTorch/torchvision, multi-floor simulation)
Date    : 2025

ARCHITECTURE:
  Layer 1 - Data Acquisition  : Camera feeds, button presses, ECU feedback
  Layer 2 - Core Processing   : VPM (YOLOv5 + Centroid Tracker), GCF, CMM
  Layer 3 - Control/Actuation : Smart Dispatch Engine (priority scoring)

WHAT'S NEW IN v2.0 vs the original report:
  - Uses torch.hub to load YOLOv5 (PyTorch-native, no subprocess needed)
  - CentroidTracker rewritten with scipy for robust Hungarian assignment
  - GCFModule and CMModule are proper classes, not bare functions
  - SmartDispatchEngine supports multi-elevator group control
  - Full simulation loop with logging and configurable parameters
  - Visual overlay renderer for camera feed debugging
=============================================================================

INSTALL (run once in your terminal):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install opencv-python scipy numpy

  # YOLOv5 weights are downloaded automatically via torch.hub on first run.
  # Internet access required on the first run only (~14 MB for yolov5s).
=============================================================================
"""

import cv2
import torch
import numpy as np
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist   # for Hungarian assignment in tracker

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("VAD")


# =============================================================================
# 1.  DATA STRUCTURES
# =============================================================================

@dataclass
class FloorCall:
    """Represents a single floor call request."""
    floor_id: int                         # Floor number (0 = ground)
    direction: str                        # "UP" | "DOWN"
    timestamp: float = field(default_factory=time.time)
    is_active: bool = True                # Turns False when GCF cancels it
    people_count: int = 0                 # Ci - validated headcount

    @property
    def wait_age(self) -> float:
        """Seconds elapsed since the button was pressed (A_i in the formula)."""
        return time.time() - self.timestamp


@dataclass
class ElevatorState:
    """Encapsulates the real-time state of one elevator car."""
    elevator_id: int
    current_floor: int = 0
    direction: str = "IDLE"              # "UP" | "DOWN" | "IDLE"
    occupancy: int = 0                   # C_elevator (people inside)
    max_capacity: int = 12               # C_max
    scheduled_stops: List[int] = field(default_factory=list)
    is_full: bool = False

    def update_full_status(self):
        self.is_full = (self.occupancy >= self.max_capacity)


# =============================================================================
# 2.  VISION PROCESSING MODULE  (Layer 2a)
#     YOLOv5 via torch.hub  +  Centroid Tracker
# =============================================================================

class CentroidTracker:
    """
    Tracks detected bounding boxes across frames by matching centroids.

    Algorithm:
      1. Compute centroid (cx, cy) for every incoming bounding box.
      2. If no existing objects → register each as new.
      3. Otherwise → build a cost matrix (Euclidean distance between all
         existing centroids and all new centroids) and use a greedy nearest-
         neighbour match (good enough; Hungarian is optional).
      4. Unmatched old objects increment their 'disappeared' counter.
         When counter > MAX_DISAPPEARED, the object is deregistered.

    Returns:
      objects  : OrderedDict  { object_id : (cx, cy) }
    """

    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        self.next_id = 0                          # Auto-incrementing unique ID
        self.objects: OrderedDict = OrderedDict() # id → (cx, cy)
        self.disappeared: Dict[int, int] = {}     # id → frames-since-seen
        self.MAX_DISAPPEARED = max_disappeared    # Frames before deregister
        self.MAX_DISTANCE = max_distance          # Pixel threshold for match

    # ------------------------------------------------------------------
    def _centroid(self, box: Tuple) -> Tuple[int, int]:
        """Convert (x1,y1,x2,y2) → integer centroid (cx, cy)."""
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    # ------------------------------------------------------------------
    def register(self, centroid: Tuple[int, int]):
        """Add a brand-new object with the next available ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        log.debug(f"[Tracker] Registered ID {self.next_id} at {centroid}")
        self.next_id += 1

    def deregister(self, obj_id: int):
        """Remove an object that has been missing too long."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        log.debug(f"[Tracker] Deregistered ID {obj_id}")

    # ------------------------------------------------------------------
    def update(self, bboxes: List[Tuple]) -> OrderedDict:
        """
        Main update call.  Pass the list of bounding boxes for the current frame.
        Returns the updated objects dictionary { id: (cx, cy) }.
        """
        # ---- Case 1: no detections ----------------------------------------
        if len(bboxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.MAX_DISAPPEARED:
                    self.deregister(obj_id)
            return self.objects

        # Compute centroids for incoming boxes
        new_centroids = [self._centroid(b) for b in bboxes]

        # ---- Case 2: no existing objects → register all -------------------
        if len(self.objects) == 0:
            for c in new_centroids:
                self.register(c)
            return self.objects

        # ---- Case 3: match new centroids to existing ones -----------------
        existing_ids = list(self.objects.keys())
        existing_cxcy = np.array(list(self.objects.values()), dtype=float)
        new_cxcy = np.array(new_centroids, dtype=float)

        # Cost matrix: distance between every existing↔new pair
        D = cdist(existing_cxcy, new_cxcy)

        # Greedy match: sort pairs by cost (smallest distance first)
        rows = D.min(axis=1).argsort()   # existing object indices sorted
        cols = D.argmin(axis=1)[rows]    # best matching new centroid index

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.MAX_DISTANCE:
                continue  # Too far away → treat as new object
            obj_id = existing_ids[row]
            self.objects[obj_id] = new_centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing → increment disappeared counter
        for row in set(range(len(existing_ids))) - used_rows:
            obj_id = existing_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.MAX_DISAPPEARED:
                self.deregister(obj_id)

        # Unmatched new → register as fresh objects
        for col in set(range(len(new_centroids))) - used_cols:
            self.register(new_centroids[col])

        return self.objects

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        """Current number of tracked unique people (C_i)."""
        return len(self.objects)


# ---------------------------------------------------------------------------

class VisionProcessingModule:
    """
    Wraps YOLOv5 (loaded via torch.hub) + CentroidTracker.

    Responsibilities:
      - Load the YOLOv5s model once at startup.
      - Accept raw BGR frames from OpenCV.
      - Apply a Region-of-Interest (ROI) mask so only the waiting area counts.
      - Run YOLOv5 inference → extract 'person' bounding boxes.
      - Feed boxes to CentroidTracker → return validated count C_i.
    """

    PERSON_CLASS_ID = 0   # COCO class 0 = "person"

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        confidence_threshold: float = 0.45,
        device: str = "auto",
    ):
        """
        Args:
            roi                  : (x1, y1, x2, y2) bounding box of the waiting area
                                   in the camera frame.  None = full frame.
            confidence_threshold : Minimum YOLO confidence to accept a detection.
            device               : "cuda", "cpu", or "auto" (picks cuda if available).
        """
        self.roi = roi
        self.conf_threshold = confidence_threshold
        self.device = self._resolve_device(device)

        log.info(f"[VPM] Loading YOLOv5s on {self.device} ...")
        # torch.hub pulls yolov5s weights (~14 MB) from GitHub on first call.
        # Set force_reload=False so subsequent runs use the cached version.
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5s",
            pretrained=True,
            force_reload=False,
            device=self.device,
            verbose=False,
        )
        self.model.conf = confidence_threshold  # Confidence threshold
        self.model.classes = [self.PERSON_CLASS_ID]  # Only detect people
        log.info("[VPM] Model loaded successfully.")

        self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ------------------------------------------------------------------
    def _apply_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Crop frame to ROI if specified; return cropped frame + offset."""
        if self.roi is None:
            return frame, (0, 0)
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2], (x1, y1)

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[int, List, np.ndarray]:
        """
        Run detection + tracking on a single BGR frame.

        Returns:
            people_count : int   → C_i  (validated unique people count)
            boxes        : list  → [(x1,y1,x2,y2), ...] in full-frame coords
            annotated    : ndarray → frame with bounding boxes drawn on it
        """
        roi_frame, (off_x, off_y) = self._apply_roi(frame)

        # Convert BGR (OpenCV) → RGB (YOLO expects RGB)
        rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

        # Run inference (returns a Results object)
        results = self.model(rgb, size=640)

        # Extract bounding boxes in xyxy format, shift back to full-frame coords
        boxes = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) == self.PERSON_CLASS_ID and conf >= self.conf_threshold:
                x1 = int(xyxy[0]) + off_x
                y1 = int(xyxy[1]) + off_y
                x2 = int(xyxy[2]) + off_x
                y2 = int(xyxy[3]) + off_y
                boxes.append((x1, y1, x2, y2))

        # Update tracker → get current C_i
        self.tracker.update(boxes)
        people_count = self.tracker.count

        # Draw annotations on the original frame
        annotated = self._draw_overlay(frame.copy(), boxes, people_count)

        return people_count, boxes, annotated

    # ------------------------------------------------------------------
    def _draw_overlay(self, frame, boxes, count):
        """Draw bounding boxes, IDs, ROI rectangle, and C_i counter."""
        # ROI rectangle
        if self.roi:
            cv2.rectangle(frame, (self.roi[0], self.roi[1]),
                          (self.roi[2], self.roi[3]), (0, 255, 255), 2)
            cv2.putText(frame, "Waiting Area (ROI)",
                        (self.roi[0], self.roi[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Bounding boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Centroid IDs
        for obj_id, (cx, cy) in self.tracker.objects.items():
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {obj_id}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # C_i counter (top-left, large red text)
        cv2.putText(frame, f"People Count (Ci): {count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        return frame


# =============================================================================
# 3.  GHOST CALL FILTER  (Layer 2b)
# =============================================================================

class GhostCallFilter:
    """
    Invalidates floor calls where the passenger has left before the elevator arrives.

    Logic (from Equation 1 in the report):
      If  Bi = True  AND  Ci = 0  for  Δt > T_ghost  →  Bi := False

    T_ghost is expressed in *frames* here (fps-independent behaviour can be
    achieved by dividing by the camera fps).

    The per-floor timer is reset to 0 as soon as a person is detected again.
    """

    def __init__(self, ghost_threshold_frames: int = 90):
        """
        Args:
            ghost_threshold_frames : Frames of empty-hall before call is cancelled.
                                     90 frames ≈ 3 seconds at 30 fps.
        """
        self.threshold = ghost_threshold_frames
        # Per-floor empty-frame counters  { floor_id → int }
        self._empty_counters: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(self, floor_id: int, call: FloorCall, people_count: int) -> bool:
        """
        Called every frame for floors with an active call.

        Returns:
            True  → call is still valid
            False → call has been classified as a Ghost Call and cancelled
        """
        if not call.is_active:
            return False  # Already cancelled

        if people_count > 0:
            # Person present → reset counter
            self._empty_counters[floor_id] = 0
            return True

        # Nobody in the waiting area
        self._empty_counters[floor_id] = self._empty_counters.get(floor_id, 0) + 1

        if self._empty_counters[floor_id] > self.threshold:
            call.is_active = False  # ← Cancel the call
            log.info(
                f"[GCF] Ghost call detected on floor {floor_id}. "
                f"Counter = {self._empty_counters[floor_id]}. CALL CANCELLED."
            )
            self._empty_counters[floor_id] = 0
            return False

        log.debug(
            f"[GCF] Floor {floor_id}: empty counter = "
            f"{self._empty_counters[floor_id]}/{self.threshold}"
        )
        return True


# =============================================================================
# 4.  CAPACITY MANAGEMENT MODULE  (Layer 2c)
# =============================================================================

class CapacityManagementModule:
    """
    Prevents the elevator from stopping to pick up passengers when it is full.

    Behaviour:
      - If C_elevator >= C_max  →  ignore external pickup calls.
      - Sends a 'FULL' signal back to the requesting floor's display.
    """

    def __init__(self):
        # Tracks which floors have been notified "FULL" already this trip
        self._notified_floors: set = set()

    # ------------------------------------------------------------------
    def check(self, elevator: ElevatorState, requesting_floor: int) -> bool:
        """
        Args:
            elevator          : Current elevator state.
            requesting_floor  : Floor that wants to board.

        Returns:
            True  → pickup is allowed
            False → elevator is full, floor display should show 'FULL'
        """
        elevator.update_full_status()

        if elevator.is_full:
            if requesting_floor not in self._notified_floors:
                log.info(
                    f"[CMM] Elevator {elevator.elevator_id} is FULL "
                    f"({elevator.occupancy}/{elevator.max_capacity}). "
                    f"Floor {requesting_floor} notified."
                )
                self._notified_floors.add(requesting_floor)
            return False  # Block pickup

        # Elevator has space → clear any prior notification for this floor
        self._notified_floors.discard(requesting_floor)
        return True

    # ------------------------------------------------------------------
    def reset_notifications(self):
        """Call after passengers alight to clear the 'full' notification set."""
        self._notified_floors.clear()


# =============================================================================
# 5.  SMART DISPATCH ENGINE  (Layer 2d)
# =============================================================================

class SmartDispatchEngine:
    """
    Calculates the Dynamic Priority Score P_i for each active floor call and
    selects the optimal next stop.

    Priority Formula (Equation 2 in the report):
        P_i = W1 * C_i  +  W2 * (1 / D_i)  +  W3 * A_i

    Variables:
        C_i  = Validated people count at floor i        (throughput driver)
        D_i  = Absolute floor distance from elevator    (efficiency driver)
        A_i  = Wait age in seconds                     (anti-starvation)
        W1, W2, W3 = Tunable weights (defaults below)

    For multi-elevator group control, the Serviceability Score S_{i,E}
    (Equation 3 in the report) is also implemented.
    """

    def __init__(
        self,
        w1: float = 10.0,  # Weight for people count (throughput)
        w2: float = 1.0,   # Weight for distance (efficiency)
        w3: float = 0.1,   # Weight for wait age (anti-starvation)
    ):
        self.W1 = w1
        self.W2 = w2
        self.W3 = w3

        self.gcf = GhostCallFilter(ghost_threshold_frames=90)
        self.cmm = CapacityManagementModule()

    # ------------------------------------------------------------------
    def priority_score(
        self,
        call: FloorCall,
        elevator: ElevatorState,
    ) -> float:
        """
        Compute P_i for a single (call, elevator) pair.
        Returns 0.0 if call is invalid (ghost or capacity exceeded).
        """
        # Gate 1: GCF already cancelled this call
        if not call.is_active:
            return 0.0

        # Gate 2: CMM says elevator is full for this floor
        if not self.cmm.check(elevator, call.floor_id):
            return 0.0

        C_i = max(call.people_count, 1)  # Avoid 0-score for unconfirmed calls
        D_i = max(abs(elevator.current_floor - call.floor_id), 1)  # avoid ÷0
        A_i = call.wait_age

        P_i = (self.W1 * C_i) + (self.W2 * (1.0 / D_i)) + (self.W3 * A_i)
        return P_i

    # ------------------------------------------------------------------
    def serviceability_score(
        self,
        call: FloorCall,
        elevator: ElevatorState,
        eta_seconds: float,
    ) -> float:
        """
        Group-control extension: S_{i,E} for assigning one elevator to one call.
        (Equation 3 in the report)

        Args:
            call          : The floor call to evaluate.
            elevator      : The candidate elevator.
            eta_seconds   : Estimated time (seconds) for this elevator to arrive.

        Returns:
            Serviceability score.  Higher = better assignment.  0 = invalid.
        """
        P_i = self.priority_score(call, elevator)
        if P_i == 0.0:
            return 0.0  # WCANCEL = 0

        # Directional bias: penalise elevators that would have to reverse
        if elevator.direction == "IDLE":
            w_dir = 1.0
        elif elevator.direction == call.direction:
            w_dir = 1.0    # Travelling same direction → no penalty
        else:
            w_dir = 0.1    # Would need to reverse → heavy penalty

        # Avoid division by zero in ETA
        safe_eta = max(eta_seconds, 0.1)

        S_i_E = 1.0 * (P_i / safe_eta) * w_dir
        return S_i_E

    # ------------------------------------------------------------------
    def select_best_stop(
        self,
        calls: List[FloorCall],
        elevator: ElevatorState,
    ) -> Optional[FloorCall]:
        """
        Single-elevator: return the FloorCall with the highest P_i.
        Returns None if no valid calls exist.
        """
        best_call, best_score = None, -1.0
        for call in calls:
            score = self.priority_score(call, elevator)
            log.debug(
                f"[SDE] Floor {call.floor_id} → P_i = {score:.2f} "
                f"(Ci={call.people_count}, D={abs(elevator.current_floor - call.floor_id)}, "
                f"Age={call.wait_age:.1f}s)"
            )
            if score > best_score:
                best_score = score
                best_call = call

        if best_call:
            log.info(
                f"[SDE] Best next stop: Floor {best_call.floor_id}  "
                f"(P_i = {best_score:.2f})"
            )
        else:
            log.info("[SDE] No valid calls. Elevator IDLE.")
        return best_call

    # ------------------------------------------------------------------
    def assign_elevators(
        self,
        calls: List[FloorCall],
        elevators: List[ElevatorState],
        eta_matrix: Dict[Tuple[int, int], float],
    ) -> Dict[int, int]:
        """
        Multi-elevator group control: assign each call to the best elevator.

        Args:
            calls       : List of active FloorCalls.
            elevators   : List of available ElevatorStates.
            eta_matrix  : {(elevator_id, floor_id) → ETA seconds}

        Returns:
            assignments : { floor_id → elevator_id }
        """
        assignments = {}
        for call in calls:
            best_elev_id, best_s = None, -1.0
            for elev in elevators:
                eta = eta_matrix.get((elev.elevator_id, call.floor_id), 60.0)
                s = self.serviceability_score(call, elev, eta)
                if s > best_s:
                    best_s = s
                    best_elev_id = elev.elevator_id
            if best_elev_id is not None:
                assignments[call.floor_id] = best_elev_id
                log.info(
                    f"[GCS] Floor {call.floor_id} → Elevator {best_elev_id}  "
                    f"(S = {best_s:.2f})"
                )
        return assignments


# =============================================================================
# 6.  VAD SYSTEM  – top-level orchestrator
# =============================================================================

class VADSystem:
    """
    Ties all modules together.

    Usage modes:
      A) Live camera: call run_live(camera_index, floor_id, elevator)
      B) Simulation : call run_simulation(num_floors, num_elevators, steps)
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        confidence: float = 0.45,
        ghost_frames: int = 90,
        max_capacity: int = 12,
        w1: float = 10.0,
        w2: float = 1.0,
        w3: float = 0.1,
    ):
        self.vpm = VisionProcessingModule(roi=roi, confidence_threshold=confidence)
        self.dispatch = SmartDispatchEngine(w1=w1, w2=w2, w3=w3)

        self.elevator = ElevatorState(
            elevator_id=0,
            current_floor=0,
            max_capacity=max_capacity,
        )
        self.active_calls: List[FloorCall] = []

    # ------------------------------------------------------------------
    # A) LIVE CAMERA MODE
    # ------------------------------------------------------------------
    def run_live(self, camera_index: int = 0, floor_id: int = 0):
        """
        Open the webcam and process frames in real-time.
        Press  'b'  to simulate a button press on the monitored floor.
        Press  'q'  to quit.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            log.error("Cannot open camera. Check camera_index.")
            return

        # Add a placeholder call for the monitored floor
        current_call = FloorCall(floor_id=floor_id, direction="DOWN")
        current_call.is_active = False  # Wait for button press
        self.active_calls.append(current_call)

        log.info("VAD Live Mode started. Press 'b' to call elevator, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- Vision pipeline ----
            people_count, _, annotated = self.vpm.process_frame(frame)

            # Update the floor call's people count
            current_call.people_count = people_count

            # ---- GCF check ----
            if current_call.is_active:
                self.dispatch.gcf.update(floor_id, current_call, people_count)

            # ---- Dispatch decision ----
            next_stop = self.dispatch.select_best_stop(
                [c for c in self.active_calls if c.is_active],
                self.elevator,
            )

            # ---- HUD overlay ----
            self._draw_hud(annotated, current_call, next_stop)
            cv2.imshow("VAD System - Floor {}".format(floor_id), annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                current_call.is_active = True
                current_call.timestamp = time.time()
                log.info(f"[UI] Button pressed on floor {floor_id}.")

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    def _draw_hud(self, frame, call, next_stop):
        """Draw status HUD on the top-right corner of the frame."""
        h, w = frame.shape[:2]
        hud_x = w - 280
        lines = [
            f"Floor call active: {call.is_active}",
            f"GCF status: {'GHOST' if not call.is_active else 'VALID'}",
            f"Elevator occupancy: {self.elevator.occupancy}/{self.elevator.max_capacity}",
            f"Next stop: {next_stop.floor_id if next_stop else 'NONE'}",
        ]
        cv2.rectangle(frame, (hud_x - 5, 5), (w - 5, 100), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (hud_x, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # B) SIMULATION MODE  (no camera required)
    # ------------------------------------------------------------------
    def run_simulation(
        self,
        num_floors: int = 10,
        simulation_steps: int = 50,
        ghost_call_probability: float = 0.3,
    ):
        """
        Text-based simulation of the full VAD pipeline.

        Each step represents one 'decision tick' (≈ 1 second real-time).

        Args:
            num_floors             : Total floors in the building.
            simulation_steps       : How many ticks to simulate.
            ghost_call_probability : Probability that a new call will be a ghost.
        """
        import random

        log.info("=" * 60)
        log.info("VAD SYSTEM SIMULATION START")
        log.info(f"Floors: {num_floors}  |  Steps: {simulation_steps}  |  "
                 f"Ghost prob: {ghost_call_probability:.0%}")
        log.info("=" * 60)

        elevator = ElevatorState(elevator_id=0, current_floor=0, max_capacity=12)
        calls: List[FloorCall] = []
        ghost_stops_prevented = 0
        total_ghost_calls = 0
        trips_taken = 0

        for step in range(simulation_steps):
            log.info(f"\n--- Step {step+1:02d} | Elevator at floor {elevator.current_floor} "
                     f"| Occupancy {elevator.occupancy}/{elevator.max_capacity} ---")

            # Randomly generate new floor calls
            if random.random() < 0.4:  # 40% chance of a new call each step
                floor = random.randint(0, num_floors - 1)
                direction = random.choice(["UP", "DOWN"])
                is_ghost = random.random() < ghost_call_probability

                # People count: 0 for ghost, 1-5 for real
                people = 0 if is_ghost else random.randint(1, 5)
                new_call = FloorCall(
                    floor_id=floor, direction=direction, people_count=people
                )
                calls.append(new_call)
                call_type = "GHOST" if is_ghost else "REAL"
                log.info(f"  New {call_type} call: Floor {floor} {direction} "
                         f"(Ci={people})")
                if is_ghost:
                    total_ghost_calls += 1

            # Run GCF on all active calls
            for call in calls:
                if call.is_active:
                    still_valid = self.dispatch.gcf.update(
                        call.floor_id, call, call.people_count
                    )
                    if not still_valid and call.people_count == 0:
                        ghost_stops_prevented += 1

            # Dispatch decision
            active_calls = [c for c in calls if c.is_active]
            next_stop = self.dispatch.select_best_stop(active_calls, elevator)

            if next_stop:
                # Move elevator one step toward next_stop
                if elevator.current_floor < next_stop.floor_id:
                    elevator.current_floor += 1
                    elevator.direction = "UP"
                elif elevator.current_floor > next_stop.floor_id:
                    elevator.current_floor -= 1
                    elevator.direction = "DOWN"
                else:
                    # Arrived at destination
                    log.info(f"  ✓ Elevator ARRIVED at floor {next_stop.floor_id}. "
                             f"Picking up {next_stop.people_count} people.")
                    elevator.occupancy = min(
                        elevator.occupancy + next_stop.people_count,
                        elevator.max_capacity,
                    )
                    elevator.update_full_status()
                    next_stop.is_active = False  # Call served
                    calls.remove(next_stop)
                    trips_taken += 1
            else:
                elevator.direction = "IDLE"

        # ---- Summary ----
        log.info("\n" + "=" * 60)
        log.info("SIMULATION COMPLETE - RESULTS SUMMARY")
        log.info("=" * 60)
        log.info(f"  Total ghost calls generated : {total_ghost_calls}")
        log.info(f"  Ghost stops prevented       : {ghost_stops_prevented}")
        gsrr = (ghost_stops_prevented / max(total_ghost_calls, 1)) * 100
        log.info(f"  Ghost Stop Reduction Rate   : {gsrr:.1f}%")
        log.info(f"  Successful trips completed  : {trips_taken}")
        log.info("=" * 60)

        return {
            "ghost_calls": total_ghost_calls,
            "ghost_stops_prevented": ghost_stops_prevented,
            "gsrr_percent": gsrr,
            "trips_completed": trips_taken,
        }
