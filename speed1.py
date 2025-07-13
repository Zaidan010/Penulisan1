from collections import defaultdict
from time import time
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

class SpeedEstimator:
    """Estimasi kecepatan kendaraan berdasarkan waktu tempuh antar garis."""

    def __init__(self, names, view_img=False, line_thickness=2, spdl_dist_thresh=10):
        self.names = names
        self.trk_history = defaultdict(list)
        self.view_img = view_img
        self.tf = line_thickness
        self.spdl = spdl_dist_thresh  # ambang batas jarak piksel dari garis
        self.spd = {}
        self.trkd_ids = []

        # Dua garis horizontal: start dan end
        self.start_line = [(20, 300), (1260, 300)]  # warna hijau
        self.end_line = [(20, 400), (1260, 400)]    # warna merah

        self.trk_pt = {}  # waktu saat melewati start line
        self.trk_pp = {}  # posisi saat melewati start line

    def estimate_speed(self, im0, tracks):
        if not tracks or tracks[0].boxes.id is None:
            return im0

        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        t_ids = tracks[0].boxes.id.int().cpu().tolist()
        annotator = Annotator(im0, line_width=self.tf)

        # Gambar garis batas
        cv2.line(im0, self.start_line[0], self.start_line[1], (0, 255, 0), self.tf * 2)   # Hijau
        cv2.line(im0, self.end_line[0], self.end_line[1], (0, 0, 255), self.tf * 2)       # Merah

        for box, t_id, cls in zip(boxes, t_ids, clss):
            track = self.trk_history[t_id]
            bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            track.append(bbox_center)

            if len(track) > 30:
                track.pop(0)

            trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            y = bbox_center[1]

            # Gambar garis lintasan kendaraan
            bbox_color = colors(int(t_id), True)
            speed_label = f"{int(self.spd[t_id])} km/h" if t_id in self.spd else self.names[int(cls)]
            annotator.box_label(box, speed_label, bbox_color)
            cv2.polylines(im0, [trk_pts], isClosed=False, color=bbox_color, thickness=self.tf)
            cv2.circle(im0, (int(bbox_center[0]), int(bbox_center[1])), self.tf * 2, bbox_color, -1)

            # Melewati garis start (hijau)
            if self.start_line[0][1] - self.spdl < y < self.start_line[0][1] + self.spdl:
                self.trk_pt[t_id] = time()
                self.trk_pp[t_id] = bbox_center

            # Melewati garis end (merah)
            if self.end_line[0][1] - self.spdl < y < self.end_line[0][1] + self.spdl:
                if t_id in self.trk_pt and t_id not in self.trkd_ids:
                    time_diff = time() - self.trk_pt[t_id]
                    if time_diff > 0:
                        pixel_dist = abs(bbox_center[1] - self.trk_pp[t_id][1])
                        meters_per_pixel = 5 / 100  # konversi kasar: 5 meter = 100 piksel
                        self.spd[t_id] = (pixel_dist * meters_per_pixel) / time_diff * 3.6  # m/s to km/h
                        self.trkd_ids.append(t_id)

        return im0


if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    names = model.names
    speed_estimator = SpeedEstimator(names)
