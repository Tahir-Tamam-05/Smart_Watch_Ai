# ================================================================
# ClassWatch — Tkinter Desktop App (Windows-friendly, Python 3.12)
# ================================================================
import os
import sys
import math
import time
import csv
import threading
import traceback
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF
import winsound  # For beep notification when distracted

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import mediapipe as mp

@dataclass
class AppConfig:
    camera_index: int = 0
    max_faces: int = 30
    min_det_conf: float = 0.5
    min_trk_conf: float = 0.5
    ear_sleep_thresh: float = 0.18
    ear_frames_sleep: int = 15
    yaw_ratio_thresh: float = 0.28
    log_interval_sec: float = 60.0
    data_dir: str = field(default_factory=lambda: os.path.join("data", "logs"))
    reports_dir: str = field(default_factory=lambda: os.path.join("data", "reports"))
    csv_path: str = field(init=False)
    preview_width: int = 960
    preview_height: int = 540
    ui_refresh_ms: int = 200
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        self.csv_path = os.path.join(self.data_dir, "attendance.csv")

FACE_ID_TO_NAME = {
    0: "user1 (11111)",
    1: "user2 (22222)"
    2: "user3 (33333)",
    3: "user4 (44444)",
    4: "user5 (55555)",
    # Add more mappings if needed
}

class EngagementTracker:
    def __init__(self, alpha: float = 0.2, max_history: int = 300):
        self.alpha = alpha
        self.value = 0.0
        self.history = deque(maxlen=max_history)
    def update(self, attentive_ratio: float) -> float:
        self.value = self.alpha * attentive_ratio + (1 - self.alpha) * self.value
        self.history.append(self.value)
        return self.value

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

def _euclid(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(pts: List[Tuple[float, float]]) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    return (_euclid(p2, p6) + _euclid(p3, p5)) / (2.0 * _euclid(p1, p4) + 1e-6)

class ClassDetector:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.cap = None
        self.mesh = None
        self.sleep_counters: Dict[int, int] = {}
        self._open_camera()
    def _open_camera(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        self.cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.preview_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.preview_height)
        time.sleep(0.1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cfg.camera_index}")
        if self.mesh is not None:
            try: self.mesh.close()
            except Exception: pass
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=self.cfg.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.min_det_conf,
            min_tracking_confidence=self.cfg.min_trk_conf
        )
    def switch_camera(self, new_index: int):
        self.cfg.camera_index = new_index
        self._open_camera()
    def read(self):
        return self.cap.read()
    def release(self):
        try:
            if self.cap is not None: self.cap.release()
        except Exception: pass
        try:
            if self.mesh is not None: self.mesh.close()
        except Exception: pass
    def _norm2pix(self, lm, w, h):
        return (int(lm.x * w), int(lm.y * h))
    def infer(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        infos: List[Dict] = []
        face_count = 0
        attentive = 0
        distracted = 0
        sleeping = 0
        if res.multi_face_landmarks:
            for idx, fl in enumerate(res.multi_face_landmarks):
                face_count += 1
                def pick(idxs): return [self._norm2pix(fl.landmark[i], w, h) for i in idxs]
                left = pick(LEFT_EYE)
                right = pick(RIGHT_EYE)
                nose_x, nose_y = self._norm2pix(fl.landmark[NOSE_TIP], w, h)
                lx, rx = left[0][0], right[3][0]
                eye_width = max(1, rx - lx)
                center_x = (lx + rx) / 2
                yaw_ratio = abs(nose_x - center_x) / eye_width
                ear_l = eye_aspect_ratio(left)
                ear_r = eye_aspect_ratio(right)
                ear = (ear_l + ear_r) / 2
                sid = idx
                counter = self.sleep_counters.get(sid, 0)
                if ear < self.cfg.ear_sleep_thresh: counter += 1
                else: counter = 0
                self.sleep_counters[sid] = counter
                if counter >= self.cfg.ear_frames_sleep:
                    status = "Sleeping"; sleeping += 1
                elif yaw_ratio > self.cfg.yaw_ratio_thresh:
                    status = "Distracted"; distracted += 1
                else:
                    status = "Attentive"; attentive += 1
                infos.append({
                    "id": sid,
                    "status": status,
                    "ear": round(float(ear), 3),
                    "yaw_ratio": round(float(yaw_ratio), 3),
                    "nose": (int(nose_x), int(nose_y)),
                })
        return infos, (face_count, attentive, distracted, sleeping), res

def ensure_csv_header(path: str):
    dirname = os.path.dirname(path)
    if dirname: os.makedirs(dirname, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "total", "attentive", "distracted", "sleeping", "engagement"])

def append_csv_row(ts: str, total: int, attn: int, dist: int, sleep: int, engagement: float, path: str):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, total, attn, dist, sleep, f"{engagement:.4f}"])

def generate_pdf_from_csv(csv_path: str, out_dir: str) -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError("No CSV found. Start detection to create logs.")
    df = pd.read_csv(csv_path)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Attendance Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    if not df.empty:
        latest = df.iloc[-1]
        pdf.cell(200, 10, f"Timestamp: {latest['timestamp']}", ln=True)
        pdf.cell(200, 10, f"Total Students: {int(latest['total'])}", ln=True)
        pdf.cell(200, 10, f"Attentive: {int(latest['attentive'])}", ln=True)
        pdf.cell(200, 10, f"Distracted: {int(latest['distracted'])}", ln=True)
        pdf.cell(200, 10, f"Sleeping: {int(latest['sleeping'])}", ln=True)
        try:
            eng_val = float(latest["engagement"])
            pdf.cell(200, 10, f"Engagement (EMA): {eng_val:.2f}", ln=True)
        except Exception: pass
    else:
        pdf.cell(200, 10, "CSV is empty. Capture some data first.", ln=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(out_path)
    return out_path

class CaptureController:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.detector = ClassDetector(cfg)
        self.eng = EngagementTracker()
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_infos: List[Dict] = []
        self.total = 0
        self.attentive = 0
        self.distracted = 0
        self.sleeping = 0
        self.engagement = 0.0
        self.last_log_time = 0.0
        ensure_csv_header(self.cfg.csv_path)
        self.thread: Optional[threading.Thread] = None
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    def switch_camera(self, index: int):
        was_running = self.running
        self.stop()
        try:
            self.detector.switch_camera(index)
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to open camera {index}:\n{e}")
        finally:
            if was_running: self.start()
    def release(self):
        self.stop()
        self.detector.release()
    def _loop(self):
        while self.running:
            try:
                ok, frame = self.detector.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                infos, totals, res = self.detector.infer(frame)
                total, attn, dist, sleep = totals

                # ---- Beep sound if any distracted ----
                if any(face["status"] == "Distracted" for face in infos):
                    try:
                        winsound.Beep(1000, 200)
                    except RuntimeError:  # Prevent crash if sound device is busy
                        pass

                ratio = (attn / max(1, total)) if total else 0.0
                eng_val = self.eng.update(ratio)
                overlay = frame.copy()
                cv2.putText(overlay, f"Total: {total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(overlay, f"Attentive: {attn}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(overlay, f"Distracted: {dist}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(overlay, f"Sleeping: {sleep}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                if res and res.multi_face_landmarks:
                    h, w = overlay.shape[:2]
                    status_dict = {info["id"]: info["status"] for info in infos}
                    for idx, fl in enumerate(res.multi_face_landmarks):
                        xs = [lm.x for lm in fl.landmark]
                        ys = [lm.y for lm in fl.landmark]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        x1, y1 = int(min_x * w), int(min_y * h)
                        x2, y2 = int(max_x * w), int(max_y * h)
                        name_str = FACE_ID_TO_NAME.get(idx, f"Face {idx}")
                        status = status_dict.get(idx, "Attentive")
                        if status == "Attentive":
                            color = (0,255,0)
                        elif status == "Sleeping":
                            color = (255,0,0)
                        elif status == "Distracted":
                            color = (0,0,255)
                        else:
                            color = (192,192,192)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(overlay, name_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                with self.lock:
                    self.latest_frame = overlay
                    self.latest_infos = infos
                    self.total = int(total)
                    self.attentive = int(attn)
                    self.distracted = int(dist)
                    self.sleeping = int(sleep)
                    self.engagement = float(eng_val)
                now = time.time()
                if total > 0 and (now - self.last_log_time) >= self.cfg.log_interval_sec:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_csv_row(ts, total, attn, dist, sleep, eng_val, self.cfg.csv_path)
                    self.last_log_time = now
                time.sleep(0.01)
            except Exception as e:
                traceback.print_exc()
                time.sleep(0.1)
    def get_snapshot(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            infos = list(self.latest_infos)
            return (frame, infos, self.total, self.attentive, self.distracted, self.sleeping, self.engagement)

class ClassWatchApp(tk.Tk):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.title("ClassWatch — Attentive / Distracted / Sleeping | Tkinter")
        self.geometry(f"{cfg.preview_width + 360}x{cfg.preview_height + 140}")
        self.minsize(960, 600)
        self.cfg = cfg
        self.ctrl = None
        self._building_ui = False
        self.preview_label = None
        self.total_var = tk.StringVar(value="0")
        self.attn_var = tk.StringVar(value="0")
        self.dist_var = tk.StringVar(value="0")
        self.sleep_var = tk.StringVar(value="0")
        self.eng_var = tk.StringVar(value="0.00")
        self.status_var = tk.StringVar(value="Status: Idle")
        self.camera_var = tk.IntVar(value=self.cfg.camera_index)
        self.tree = None
        self.start_btn = None
        self.stop_btn = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            self.ctrl = CaptureController(self.cfg)
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.status_var.set(f"Status: Error — {e}")
            self.ctrl = None
        self.after(self.cfg.ui_refresh_ms, self._refresh_ui)
    def _build_ui(self):
        self._building_ui = True
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, textvariable=self.status_var).pack(side="left")
        right_controls = ttk.Frame(top)
        right_controls.pack(side="right")
        ttk.Label(right_controls, text="Camera Index:").pack(side="left", padx=(0, 6))
        cam_entry = ttk.Spinbox(right_controls, from_=0, to=9, width=5, textvariable=self.camera_var)
        cam_entry.pack(side="left")
        switch_btn = ttk.Button(right_controls, text="Switch", command=self._switch_camera)
        switch_btn.pack(side="left", padx=(6, 0))
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.preview_label = ttk.Label(left)
        self.preview_label.pack(fill="both", expand=True)
        stats = ttk.Frame(left)
        stats.pack(fill="x", pady=8)
        def stat_card(parent, title, var):
            fr = ttk.LabelFrame(parent, text=title)
            fr.pack(side="left", padx=6)
            lab = ttk.Label(fr, textvariable=var, font=("Segoe UI", 16, "bold"))
            lab.pack(padx=10, pady=6)
        stat_card(stats, "Students", self.total_var)
        stat_card(stats, "Attentive", self.attn_var)
        stat_card(stats, "Distracted", self.dist_var)
        stat_card(stats, "Sleeping", self.sleep_var)
        stat_card(stats, "Engagement (EMA)", self.eng_var)
        right = ttk.Frame(main, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        controls = ttk.LabelFrame(right, text="Controls")
        controls.pack(fill="x", padx=0, pady=(0, 8))
        self.start_btn = ttk.Button(controls, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=6, pady=6)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6, pady=6)
        save_btn = ttk.Button(controls, text="Save Snapshot", command=self._save_snapshot)
        save_btn.pack(side="left", padx=6, pady=6)
        pdf_btn = ttk.Button(controls, text="Generate PDF", command=self._make_pdf)
        pdf_btn.pack(side="left", padx=6, pady=6)
        settings = ttk.LabelFrame(right, text="Detection Settings")
        settings.pack(fill="x", pady=(0, 8))
        ear_row = ttk.Frame(settings)
        ear_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(ear_row, text="EAR sleep thresh:").pack(side="left")
        self.ear_var = tk.DoubleVar(value=self.cfg.ear_sleep_thresh)
        ear_spin = ttk.Spinbox(ear_row, from_=0.10, to=0.40, increment=0.01, textvariable=self.ear_var, width=6, command=self._apply_settings)
        ear_spin.pack(side="left", padx=6)
        frames_row = ttk.Frame(settings)
        frames_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(frames_row, text="Frames for sleeping:").pack(side="left")
        self.sleep_frames_var = tk.IntVar(value=self.cfg.ear_frames_sleep)
        frames_spin = ttk.Spinbox(frames_row, from_=5, to=60, increment=1, textvariable=self.sleep_frames_var, width=6, command=self._apply_settings)
        frames_spin.pack(side="left", padx=6)
        yaw_row = ttk.Frame(settings)
        yaw_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(yaw_row, text="Yaw ratio thresh:").pack(side="left")
        self.yaw_var = tk.DoubleVar(value=self.cfg.yaw_ratio_thresh)
        yaw_spin = ttk.Spinbox(yaw_row, from_=0.10, to=0.60, increment=0.01, textvariable=self.yaw_var, width=6, command=self._apply_settings)
        yaw_spin.pack(side="left", padx=6)
        table = ttk.LabelFrame(right, text="Faces (live)")
        table.pack(fill="both", expand=True)
        cols = ("id", "status", "ear", "yaw")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", height=14)
        self.tree.heading("id", text="ID")
        self.tree.heading("status", text="Status")
        self.tree.heading("ear", text="EAR")
        self.tree.heading("yaw", text="Yaw")
        self.tree.column("id", width=40, anchor="center")
        self.tree.column("status", width=100, anchor="w")
        self.tree.column("ear", width=70, anchor="e")
        self.tree.column("yaw", width=70, anchor="e")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)
        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(footer, text="CSV:").pack(side="left")
        ttk.Label(footer, text=self.cfg.csv_path, foreground="gray").pack(side="left", padx=(6, 0))
        self._building_ui = False
    def _apply_settings(self):
        if self._building_ui or self.ctrl is None: return
        try:
            self.ctrl.detector.cfg.ear_sleep_thresh = float(self.ear_var.get())
            self.ctrl.detector.cfg.ear_frames_sleep = int(self.sleep_frames_var.get())
            self.ctrl.detector.cfg.yaw_ratio_thresh = float(self.yaw_var.get())
        except Exception as e:
            messagebox.showwarning("Settings", f"Could not apply settings:\n{e}")
    def _switch_camera(self):
        if self.ctrl is None: return
        idx = int(self.camera_var.get())
        self.status_var.set(f"Status: Switching camera → {idx} ...")
        self.update_idletasks()
        self.ctrl.switch_camera(idx)
        self.status_var.set("Status: Ready")
    def _start(self):
        if self.ctrl is None:
            try:
                self.ctrl = CaptureController(self.cfg)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status_var.set(f"Status: Error — {e}")
                return
        self.ctrl.start()
        self.status_var.set("Status: Running")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
    def _stop(self):
        if self.ctrl: self.ctrl.stop()
        self.status_var.set("Status: Stopped")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    def _save_snapshot(self):
        if self.ctrl is None:
            messagebox.showinfo("Snapshot", "Start the camera first."); return
        frame, *_ = self.ctrl.get_snapshot()
        if frame is None:
            messagebox.showinfo("Snapshot", "No frame available yet."); return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"snapshot_{ts}.jpg"
        path = filedialog.asksaveasfilename(
            title="Save snapshot as...",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"), ("All Files", "*.*")]
        )
        if not path: return
        try:
            cv2.imwrite(path, frame)
            messagebox.showinfo("Snapshot", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Snapshot", f"Failed to save:\n{e}")
    def _make_pdf(self):
        try:
            out = generate_pdf_from_csv(self.cfg.csv_path, self.cfg.reports_dir)
            messagebox.showinfo("Report", f"PDF saved:\n{out}")
        except Exception as e:
            messagebox.showerror("Report", f"Failed to generate PDF:\n{e}")
    def _refresh_ui(self):
        try:
            if self.ctrl is not None:
                frame, infos, total, attn, dist, sleep, eng = self.ctrl.get_snapshot()
                self.total_var.set(str(total))
                self.attn_var.set(str(attn))
                self.dist_var.set(str(dist))
                self.sleep_var.set(str(sleep))
                try: self.eng_var.set(f"{float(eng):.2f}")
                except Exception: self.eng_var.set(str(eng))
                self._update_table(infos)
                if frame is not None:
                    disp = self._prepare_display_image(frame, self.cfg.preview_width, self.cfg.preview_height)
                    self.preview_label.configure(image=disp)
                    self.preview_label.image = disp
        except Exception:
            traceback.print_exc()
        self.after(self.cfg.ui_refresh_ms, self._refresh_ui)
    def _update_table(self, faces: List[Dict]):
        existing = {self.tree.set(iid, "id"): iid for iid in self.tree.get_children("")}
        current_ids = set(str(f["id"]) for f in faces)
        for iid in list(self.tree.get_children("")):
            if self.tree.set(iid, "id") not in current_ids:
                self.tree.delete(iid)
        for f in faces:
            fid = str(f["id"])
            status = f.get("status", "")
            ear = f.get("ear", "")
            yaw = f.get("yaw_ratio", "")
            if fid in existing:
                iid = existing[fid]
                self.tree.set(iid, "status", status)
                self.tree.set(iid, "ear", ear)
                self.tree.set(iid, "yaw", yaw)
            else:
                iid = self.tree.insert("", "end", values=(fid, status, ear, yaw))
                existing[fid] = iid
    @staticmethod
    def _prepare_display_image(frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image=img)
    def _on_close(self):
        try:
            if self.ctrl: self.ctrl.release()
        except Exception: pass
        self.destroy()

def main():
    cfg = AppConfig()
    app = ClassWatchApp(cfg)
    app.mainloop()

if __name__ == "__main__":
    main()
