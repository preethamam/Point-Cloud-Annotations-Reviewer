#!/usr/bin/env python3
# Point Cloud Reviewer — PyQt5 + VisPy (fast 3-D, synced dual view)
# Features: folders dialog (persisted), per-scene comments (LocalAppData), Excel export,
# overlay toggle (red-on-blue vs as-is), compact right-justified slider, 1/N counter,
# shared 3-D camera (rotate/pan/zoom) for left & right canvases, Save PNG (composite).

import os, re, glob, sys, time, json, shutil
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import open3d as o3d
from tqdm import tqdm
import pandas as pd            # pip install pandas openpyxl
from PIL import Image          # pip install pillow

# ==== NEW: fast GPU renderer (VisPy) ==========================================
from vispy import scene
from vispy.scene import visuals
# ==============================================================================

# ======= TUNABLES ==============================================================
APP_NAME               = "Point Cloud Reviewer"
TARGET_W_PX, TARGET_H_PX = 1100, 650
ALLOWED_EXT            = (".ply", ".pcd")
MAX_PTS                = 2_000_000     # interactive decimation cap
COMMENT_BOX_HEIGHT     = 40            # tweak comment box height here
SLIDER_WIDTH_PX        = 180           # compact slider width (pixels)
MARKER_SCALE_VISPY     = 1.0           # VisPy size = slider_value * this factor
ZOOM_BASE              = 1.25          # 1.25–1.6 is a good range; increase for faster per-tick zoom
# ==============================================================================

LOCALAPP = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData/Local")
APP_DIR  = os.path.join(LOCALAPP, APP_NAME)
SETTINGS = os.path.join(APP_DIR, "settings.json")
COMMENTS = os.path.join(APP_DIR, "comments.json")

stem      = lambda p: os.path.splitext(os.path.basename(p))[0]
file_stem = lambda p: os.path.basename(p)
natural_k = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
ensure_dir= lambda p: os.makedirs(p, exist_ok=True)

def load_settings() -> dict:
    ensure_dir(APP_DIR)
    if not os.path.exists(SETTINGS):
        data = {
            "ORIG_DIRS": [],
            "ANNO_DIRS": [],
            "REVISE_DIRS": [],
            "last_export_path": str(Path.home()),
            "overlay_red_on_blue": True,
            "continue_where_left": True,
            "last_stem": "",
            "last_index": 0
        }
        with open(SETTINGS, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        return data
    with open(SETTINGS, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except Exception: return {"ORIG_DIRS": [], "ANNO_DIRS": [], "REVISE_DIRS": [], "overlay_red_on_blue": True}

def save_settings(data: dict):
    ensure_dir(APP_DIR)
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    os.replace(tmp, SETTINGS)

def ensure_comments() -> Dict[str, str]:
    ensure_dir(APP_DIR)
    if not os.path.exists(COMMENTS):
        with open(COMMENTS, "w", encoding="utf-8") as f: json.dump({}, f, indent=2)
    with open(COMMENTS, "r", encoding="utf-8") as f:
        try: return json.load(f) or {}
        except Exception: return {}

def save_comments(data: Dict[str, str]):
    tmp = COMMENTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    os.replace(tmp, COMMENTS)

def find_file(st: str, roots: List[str]) -> Optional[str]:
    for r in roots:
        for ext in ALLOWED_EXT:
            p = os.path.join(r, st + ext)
            if os.path.exists(p): return p
    return None

def load_pc(p: Optional[str]) -> Optional[o3d.geometry.PointCloud]:
    return o3d.io.read_point_cloud(p) if p else None

def pc_to_xyz_rgb(pc: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N,3) xyz and (N,3) rgb; decimate to MAX_PTS if needed."""
    pts = np.asarray(pc.points)
    col = np.asarray(pc.colors)
    if len(pts) > MAX_PTS:
        idx = np.random.choice(len(pts), MAX_PTS, replace=False)
        pts, col = pts[idx], col[idx]
    return pts, col

def red_mask(c: np.ndarray) -> np.ndarray:
    return (c[:,0] >= 0.8) & (c[:,0] > c[:,1] + 0.1) & (c[:,0] > c[:,2] + 0.1)

def _safe_print(*a, **k):
    try:
        print(*a, **k)
    except Exception:
        pass
    
# --- add near imports (after tqdm import is fine) ---
class _DummyTQDM:
    def __init__(self, total=1, initial=0, **kw):
        self.total = int(total)
        self.n = int(initial)
    def update(self, n=1): self.n += int(n)
    def refresh(self): pass
    def close(self):   pass

def _make_pbar(total, *, initial=0, **kw):
    """Return real tqdm if a TTY is present; otherwise a no-op bar."""
    try:
        is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    except Exception:
        is_tty = False
    if is_tty:
        # use tqdm with your preferred formatting
        return tqdm(total=total, initial=initial, **kw)
    else:
        return _DummyTQDM(total=total, initial=initial)

# ---------------- Folder dialog ----------------
class FolderPrefsDialog(QtWidgets.QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Folders")
        self.resize(900, 450)
        self.settings = settings
        grid = QtWidgets.QGridLayout(self)

        def make_list(paths):
            lw = QtWidgets.QListWidget(); lw.addItems(paths)
            lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            return lw

        self.orig = make_list(settings.get("ORIG_DIRS", []))
        self.anno = make_list(settings.get("ANNO_DIRS", []))
        self.revi = make_list(settings.get("REVISE_DIRS", []))

        grid.addWidget(QtWidgets.QLabel("Original Folders"),   0,0)
        grid.addWidget(QtWidgets.QLabel("Annotation Folders"), 0,1)
        grid.addWidget(QtWidgets.QLabel("Revise Folders"),     0,2)
        grid.addWidget(self.orig, 1,0); grid.addWidget(self.anno,1,1); grid.addWidget(self.revi,1,2)

        def controls(lw):
            h=QtWidgets.QHBoxLayout()
            add=QtWidgets.QPushButton("Add"); rm=QtWidgets.QPushButton("Remove"); up=QtWidgets.QPushButton("↑"); dn=QtWidgets.QPushButton("↓")
            h.addWidget(add); h.addWidget(rm); h.addWidget(up); h.addWidget(dn)
            def add_path():
                dlg = QtWidgets.QFileDialog(self, "Select Folders")
                dlg.setFileMode(QtWidgets.QFileDialog.Directory)
                dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)  # required for multi-select
                dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)

                # Allow multi-selection in the dialog's internal views
                for view in dlg.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                    view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

                paths = []
                if dlg.exec_():
                    # Try the standard API first
                    paths = [p for p in dlg.selectedFiles() if p]

                    # If Qt only returned the "current" dir, pull all explicitly selected dirs
                    if len(paths) <= 1:
                        root = dlg.directory()
                        seen = set()
                        for view in dlg.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                            sm = view.selectionModel()
                            if not sm:
                                continue
                            for idx in sm.selectedIndexes():
                                if idx.column() != 0:
                                    continue
                                p = root.absoluteFilePath(idx.data())
                                if os.path.isdir(p) and p not in seen:
                                    seen.add(p)
                                    paths.append(p)

                # add (de-duped, case-insensitive on Windows)
                existing_norm = {os.path.normcase(os.path.normpath(lw.item(i).text()))
                                for i in range(lw.count())}
                for p in paths:
                    p_norm = os.path.normcase(os.path.normpath(os.path.abspath(p)))
                    if p_norm not in existing_norm:
                        lw.addItem(p)
                        existing_norm.add(p_norm)  # avoid dupes within the same batch

            def remove_sel():
                for it in lw.selectedItems(): lw.takeItem(lw.row(it))
                
            def move(delta):
                rows=sorted([lw.row(it) for it in lw.selectedItems()])
                if not rows: return
                if delta<0 and rows[0]==0: return
                if delta>0 and rows[-1]==lw.count()-1: return
                for r in rows:
                    it=lw.takeItem(r); lw.insertItem(r+delta,it); lw.setItemSelected(it,True)
            add.clicked.connect(add_path); rm.clicked.connect(remove_sel)
            up.clicked.connect(lambda: move(-1)); dn.clicked.connect(lambda: move(+1))
            return h

        grid.addLayout(controls(self.orig), 2,0)
        grid.addLayout(controls(self.anno), 2,1)
        grid.addLayout(controls(self.revi), 2,2)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save|QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        grid.addWidget(btns, 3,0,1,3)

    def values(self) -> dict:
        def collect(lw): return [lw.item(i).text() for i in range(lw.count())]
        return {"ORIG_DIRS": collect(self.orig), "ANNO_DIRS": collect(self.anno), "REVISE_DIRS": collect(self.revi)}

# ---------------- VisPy dual canvas (shared camera) -----------------
class DualCanvasVispy(QtWidgets.QWidget):
    """
    Two side-by-side VisPy canvases with a SHARED TurntableCamera.
    Left = Original, Right = Annotation (overlay/as-is).
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.left  = scene.SceneCanvas(keys=None, size=(600, 600), bgcolor='white', show=False)
        self.right = scene.SceneCanvas(keys=None, size=(600, 600), bgcolor='white', show=False)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)

        # left panel with title
        left_container = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_container)
        left_v.setContentsMargins(0,0,0,0)
        self.titleL = QtWidgets.QLabel("Original")
        self.titleL.setAlignment(QtCore.Qt.AlignHCenter)
        self.titleL.setStyleSheet("font-weight:600;")
        left_v.addWidget(self.titleL)
        left_v.addWidget(self.left.native, 1)

        # right panel with title
        right_container = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right_container)
        right_v.setContentsMargins(0,0,0,0)
        self.titleR = QtWidgets.QLabel("Annotation")
        self.titleR.setAlignment(QtCore.Qt.AlignHCenter)
        self.titleR.setStyleSheet("font-weight:600;")
        right_v.addWidget(self.titleR)
        right_v.addWidget(self.right.native, 1)

        lay.addWidget(left_container, 1)
        lay.addWidget(right_container, 1)

        # views
        self.vl = self.left.central_widget.add_view()
        self.vr = self.right.central_widget.add_view()

        # --- two cameras (no shared camera to avoid transform drift) ---
        self.camL = scene.cameras.TurntableCamera(fov=0.0, azimuth=0.0, elevation=90.0, up='+z', distance=1.0)
        self.camR = scene.cameras.TurntableCamera(fov=0.0, azimuth=0.0, elevation=90.0, up='+z', distance=1.0)
        self.vl.camera = self.camL
        self.vr.camera = self.camR

        # start both with the same state
        self.camR.set_state(self.camL.get_state())

        # --- take over the wheel on each view (prevent double-zoom) ---
        for v in (self.vl, self.vr):
            try:
                v.events.mouse_wheel.disconnect(v.camera.on_mouse_wheel)  # stop default zoom
            except Exception:
                pass
            # bind our handler; capture the view in the lambda
            v.events.mouse_wheel.connect(lambda ev, vv=v: self._on_wheel(ev, vv))

        # --- keep cameras mirrored during pan/rotate/drag (no camera event names needed) ---
        # mirror from whichever view is being interacted with
        self.left.events.mouse_move.connect(   lambda ev, vv=self.vl: self._mirror_from_view(vv))
        self.right.events.mouse_move.connect(  lambda ev, vv=self.vr: self._mirror_from_view(vv))
        self.left.events.mouse_release.connect(lambda ev, vv=self.vl: self._mirror_from_view(vv))
        self.right.events.mouse_release.connect(lambda ev, vv=self.vr: self._mirror_from_view(vv))

        # handles + cached data
        self.h_left = self.h_right_base = self.h_right_overlay = None
        self._xyzL = self._xyzRB = self._xyzRO = None      # cached positions for size updates
        self._rgbL = self._rgbRB = self._rgbRO = None
        
        self._pt_size = 1.0                      # <-- track current marker size
        self.overlay_alpha = 1.0                 # <-- NEW: overlay transparency (0..1)


    @staticmethod
    def _markers(xyz: np.ndarray, rgb: np.ndarray, size: float, *, depth_test=True, alpha: float = 1.0) -> visuals.Markers:
        m = visuals.Markers()
        if xyz is None or len(xyz) == 0:
            return m
        if rgb is None or len(rgb) == 0:
            rgb = np.ones((len(xyz), 3), dtype=np.float32)
        a = float(np.clip(alpha, 0.0, 1.0))
        rgba = np.c_[np.clip(rgb,0,1), np.full((len(rgb),1), a, dtype=np.float32)]
        m.set_data(pos=xyz.astype(np.float32),
                face_color=rgba.astype(np.float32),
                edge_width=0, size=float(size))
        # draw-state: blend, and (optionally) disable depth test
        m.set_gl_state('opaque' if depth_test else 'translucent', depth_test=bool(depth_test))
        m.antialias = 0
        return m

    def _set_range_from_bounds(self, xyz):
        if xyz is None or len(xyz) == 0:
            return
        mn = xyz.min(axis=0); mx = xyz.max(axis=0)
        self._xyz_bounds = (mn, mx)
        center = (mn + mx) * 0.5
        extent = float(np.linalg.norm(mx - mn)) or 1.0

        # set left, then mirror to right via the same state
        self.camL.center = center
        self.camL.distance = extent * 1.2
        # push the same state to the right (without recursion)
        self._syncing = True
        self.camR.set_state(self.camL.get_state())
        self._syncing = False

    def set_titles(self, left: str, right: str):
        self.titleL.setText(left)
        self.titleR.setText(right)

    def clear(self):
        for h in (self.h_left, self.h_right_base, self.h_right_overlay):
            if h is not None and h.parent is not None:
                h.parent = None
        self.h_left = self.h_right_base = self.h_right_overlay = None

    def set_left(self, xyz: np.ndarray, rgb: np.ndarray, size: float):
        self._xyzL, self._rgbL = xyz, rgb
        self._pt_size = float(size)
        if xyz is None or len(xyz) == 0:
            self.h_left = None; return
        self.h_left = self._markers(xyz, rgb, size); self.vl.add(self.h_left)

    def set_right_base(self, xyz: np.ndarray, rgb: np.ndarray, size: float):
        self._xyzRB, self._rgbRB = xyz, rgb
        self._pt_size = float(size)
        if xyz is None or len(xyz) == 0:
            self.h_right_base = None; return
        self.h_right_base = self._markers(xyz, rgb, size, depth_test=True)   # unchanged depth
        self.vr.add(self.h_right_base)

    def set_right_overlay(self, xyz: Optional[np.ndarray], rgb: Optional[np.ndarray], size: float):
        self._xyzRO, self._rgbRO = xyz, rgb
        if xyz is None or len(xyz) == 0:
            self.h_right_overlay = None; return
        self.h_right_overlay = self._markers(xyz, rgb, size, depth_test=False, alpha=self.overlay_alpha)
        self.vr.add(self.h_right_overlay)

    def set_point_size(self, size: float):
        def apply(h, xyz, rgb):
            if h is None or xyz is None or rgb is None: 
                return
            rgba = np.c_[np.clip(rgb, 0, 1), np.ones((len(rgb), 1))].astype(np.float32)
            h.set_data(pos=xyz.astype(np.float32), face_color=rgba, edge_width=0, size=float(size))
            h.antialias = 0

        apply(self.h_left,         self._xyzL,  self._rgbL)
        apply(self.h_right_base,   self._xyzRB, self._rgbRB)
        apply(self.h_right_overlay,self._xyzRO, self._rgbRO)
    
    def set_overlay_alpha(self, alpha: float):
        """Set transparency (0..1) for the RED overlay layer only."""
        self.overlay_alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.h_right_overlay is None or self._xyzRO is None or self._rgbRO is None:
            return
        rgba = np.c_[np.clip(self._rgbRO, 0, 1), np.full((len(self._rgbRO), 1), self.overlay_alpha, dtype=np.float32)].astype(np.float32)
        self.h_right_overlay.set_data(
            pos=self._xyzRO.astype(np.float32),
            face_color=rgba,
            edge_width=0,
            size=float(self._pt_size)
        )

    def _fit_bounds_top(self, xyz):
        if xyz is None or len(xyz) == 0: return
        mn = xyz.min(axis=0); mx = xyz.max(axis=0)
        center = (mn + mx) * 0.5
        extent = float(np.linalg.norm(mx - mn)) or 1.0

        # top view + orthographic scale fit
        self.camL.center = center
        self.camL.azimuth = 0.0
        self.camL.elevation = 90.0
        self.camL.roll = 0.0
        self.camL.scale_factor = extent * 0.8  # <-- use scale_factor for fov=0 (ortho)

        self._syncing = True
        self.camR.set_state(self.camL.get_state())
        self._syncing = False

    def fit_to_data_top(self):
        # choose the biggest available cloud (left preferred, then right base)
        xyz = self._xyzL
        if xyz is None or len(xyz) == 0:
            xyz = self._xyzRB
        self._fit_bounds_top(xyz)

    def reset_view(self):
        self.fit_to_data_top()          
        
    # Composite screenshot of both canvases, side-by-side
    def screenshot_rgba(self) -> np.ndarray:
        imgL = self.left.render()   # RGBA uint8
        imgR = self.right.render()
        h = max(imgL.shape[0], imgR.shape[0])
        # pad shorter one to match height
        def pad_to_h(img):
            if img.shape[0] == h: return img
            pad = np.zeros((h - img.shape[0], img.shape[1], 4), dtype=img.dtype)
            return np.vstack([img, pad])
        return np.hstack([pad_to_h(imgL), pad_to_h(imgR)])    
    
    def _mirror_from_view(self, src_view):
        """Copy full camera state from src view to the other view."""
        src_cam = src_view.camera
        dst_cam = self.camR if src_view is self.vl else self.camL
        try:
            dst_cam.set_state(src_cam.get_state())
        except Exception:
            pass

    def _ray_through_canvas_xy(self, view, x, y):
        """World-space ray through pixel (x,y) for THIS view (canvas -> scene)."""
        scn_to_can = view.scene.node_transform(view.canvas.scene)
        can_to_scn = scn_to_can.inverse
        p0 = can_to_scn.map([float(x), float(y), 0.0, 1.0])  # near
        p1 = can_to_scn.map([float(x), float(y), 1.0, 1.0])  # far
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        p0 = p0[:3] / p0[3] if p0.shape[-1] == 4 and abs(p0[3]) > 1e-12 else p0[:3]
        p1 = p1[:3] / p1[3] if p1.shape[-1] == 4 and abs(p1[3]) > 1e-12 else p1[:3]
        d = p1 - p0; n = float(np.linalg.norm(d)); d = d / max(n, 1e-12)
        return p0, d

    def _on_wheel(self, ev, view):
        """AutoCAD-style zoom anchored at cursor (fast + drift-free) for the active view."""
        try:
            ev.handled = True
        except Exception:
            pass

        # --- normalize wheel steps (some backends give ±120, some give ±1) ---
        try:
            raw = float(getattr(ev, "delta", (0, 0))[1]) if hasattr(ev, "delta") else float(ev.delta[1])
        except Exception:
            return
        if raw == 0:
            return
        steps = raw/120.0 if abs(raw) >= 40.0 else raw  # >=40 ⇒ legacy ±120; else already in steps

        # --- choose zoom base and apply keyboard modifiers ---
        base = float(globals().get("ZOOM_BASE", 1.45))
        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
            if mods & QtCore.Qt.ControlModifier:
                base = base ** 3      # coarse
            elif mods & QtCore.Qt.ShiftModifier:
                base = base ** (1/3)  # fine
        except Exception:
            pass

        factor = base ** steps  # >1 zoom in, <1 zoom out

        cam = view.camera  # this view's camera (we have one per view)
        w, h = view.size
        if w <= 0 or h <= 0:
            return
        x, y = float(ev.pos[0]), float(ev.pos[1])

        # --- plane: pre-zoom camera-center plane, normal = center ray direction ---
        oc, nc = self._ray_through_canvas_xy(view, 0.5 * w, 0.5 * h)  # nc is unit
        c0 = np.array(cam.center, float)

        # anchor BEFORE zoom
        o0, d0 = self._ray_through_canvas_xy(view, x, y)
        denom0 = float(np.dot(d0, nc))
        if abs(denom0) < 1e-12:
            return
        t0 = float(np.dot(c0 - o0, nc) / denom0)
        anchor = o0 + d0 * t0

        # --- apply zoom (ortho path; perspective fallback supported) ---
        if (getattr(cam, "fov", 0.0) or 0.0) == 0.0:
            cam.scale_factor = float(cam.scale_factor) / max(factor, 1e-6)   # orthographic
        else:
            cam.distance     = float(cam.distance)     / max(factor, 1e-6)   # perspective

        # anchor AFTER zoom (same pre-zoom plane)
        o1, d1 = self._ray_through_canvas_xy(view, x, y)
        denom1 = float(np.dot(d1, nc))
        if abs(denom1) < 1e-12:
            return
        t1 = float(np.dot(c0 - o1, nc) / denom1)
        q1 = o1 + d1 * t1

        # pan to keep anchor pinned
        pan = anchor - q1
        if np.all(np.isfinite(pan)):
            cam.center = (float(c0[0] + pan[0]), float(c0[1] + pan[1]), float(c0[2] + pan[2]))

        # mirror to the other camera and refresh both
        self._mirror_from_view(view)
        try:
            self.left.update(); self.right.update()
        except Exception:
            pass

# ---------------- Main window -------------------
class ReviewerApp(QtWidgets.QMainWindow):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        
        # Track revised files across the session (and persist them)
        self.settings.setdefault("REVISED_FILES", [])
        self.settings.setdefault("REVISED_ORIGS", {})   # map revised anno path -> absolute original path

        # Clean out any non-existent paths
        self.settings["REVISED_FILES"] = [p for p in self.settings["REVISED_FILES"] if os.path.exists(p)]
        save_settings(self.settings)

        self.ORIG_DIRS   = settings.get("ORIG_DIRS", [])
        self.ANNO_DIRS   = settings.get("ANNO_DIRS", [])
        self.REVISE_DIRS = settings.get("REVISE_DIRS", [])
        self.overlay_mode= bool(settings.get("overlay_red_on_blue", True))
        self.last_export = settings.get("last_export_path", str(Path.home()))

        self.comments: Dict[str,str] = ensure_comments()
        self.stems, self.anno_map, self.orig_map = self.collect()    
        
        # --- one-time migration from legacy stem-only keys (only when unambiguous) ---
        def _migrate_legacy_comments():
            # count how many times each stem_only appears across current list
            stem_count = {}
            for k in self.stems:
                ap = self.anno_map.get(k, "")
                st = os.path.splitext(os.path.basename(ap or ""))[0]
                stem_count[st] = stem_count.get(st, 0) + 1

            changed = False
            for k in self.stems:
                if k in self.comments:
                    continue
                ap = self.anno_map.get(k, "")
                st = os.path.splitext(os.path.basename(ap or ""))[0]
                # migrate only if this stem appears exactly once (no ambiguity)
                if stem_count.get(st, 0) == 1 and st in self.comments:
                    self.comments[k] = self.comments.pop(st)
                    changed = True
            if changed:
                save_comments(self.comments)

        _migrate_legacy_comments()
      
        self.idx=0; self.total=len(self.stems)
        self._seen: set = set()
        if self.total > 0:
            # mark the first item as already visited
            self._seen.add(self.stems[self.idx])
        self.point_size = 5.0    # slider shows this

        # jump to last position if enabled and available
        if self.total and self.settings.get("continue_where_left", True):
            last_stem  = self.settings.get("last_stem", "")
            last_index = self.settings.get("last_index", 0)

            # Priority 1: exact stem match (most reliable if filenames are same)
            if last_stem in self.stems:
                self.idx = self.stems.index(last_stem)
            # Priority 2: numeric index fallback
            elif 0 <= last_index < self.total:
                self.idx = last_index
            else:
                self.idx = 0  # safe fallback

        # UI setup
        self.setWindowTitle(APP_NAME)
        self.resize(TARGET_W_PX, TARGET_H_PX+220)
        icon_path = os.path.join(os.getcwd(), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        central=QtWidgets.QWidget(self); self.setCentralWidget(central)
        v=QtWidgets.QVBoxLayout(central); v.setContentsMargins(6,6,6,6); v.setSpacing(6)

        # top bar
        top=QtWidgets.QHBoxLayout()
        self.btn_folders   = QtWidgets.QPushButton("Folders")
        self.chk_overlay   = QtWidgets.QCheckBox("Overlay on Original PC (O)")
        self.chk_overlay.setChecked(self.overlay_mode)
        
        # --- NEW: show/hide red annotations toggle (A) ---
        self.show_annotations = True
        self.chk_show_ann = QtWidgets.QCheckBox("Toggle Annotations (A)")
        self.chk_show_ann.setChecked(self.show_annotations)
        
        # NEW: continue-where-left
        self.chk_resume = QtWidgets.QCheckBox("Continue where you left")
        self.chk_resume.setChecked(bool(self.settings.get("continue_where_left", True)))

        self.btn_export_xl = QtWidgets.QPushButton("Export Excel")
        top.addWidget(self.btn_folders)
        top.addStretch(1)
        top.addWidget(self.chk_overlay)
        top.addSpacing(10)
        top.addWidget(self.chk_show_ann)
        top.addSpacing(10)
        top.addWidget(self.chk_resume)
        top.addSpacing(10)
        top.addWidget(self.btn_export_xl)
        v.addLayout(top)

        # canvases (VisPy)
        self.canvas = DualCanvasVispy(self); v.addWidget(self.canvas, 1)

        # bottom controls
        grid=QtWidgets.QGridLayout(); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(4)
        v.addLayout(grid, 0)

        self.lbl_comment=QtWidgets.QLabel("Comment (this scene):")
        self.txt_comment=QtWidgets.QPlainTextEdit(); self.txt_comment.setPlaceholderText("Type review notes…")
        self.txt_comment.setFixedHeight(COMMENT_BOX_HEIGHT)

        self.btn_prev = QtWidgets.QPushButton("Previous")
        self.btn_next = QtWidgets.QPushButton("Next")
        self.btn_revise= QtWidgets.QPushButton("Revise (m/M)")
        self.btn_png  = QtWidgets.QPushButton("Save PNG (Ctrl+S)")
        self.btn_savec= QtWidgets.QPushButton("Save Comment")
        self.btn_reset = QtWidgets.QPushButton("Reset View (r/R)")
        self.btn_savec.setFixedWidth(self.btn_prev.sizeHint().width()+8)
        self.btn_savec.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.lbl_ps   = QtWidgets.QLabel("Point Size:")
        self.sld_ps   = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_ps.setRange(1, 80)
        self.sld_ps.setSingleStep(1)
        self.sld_ps.setValue(int(self.point_size*10))
        self.sld_ps.setFixedWidth(SLIDER_WIDTH_PX)
        self.lbl_ps_val = QtWidgets.QLabel(f"{self.point_size:.1f}")
        self.lbl_ps.setContentsMargins(0,0,6,0)
        
        # --- NEW: Transparency slider (affects red overlay only) ---
        self.lbl_alpha = QtWidgets.QLabel("Transparency:")
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)                 # 0%..100%
        self.sld_alpha.setSingleStep(1)
        self.sld_alpha.setValue(100)                    # default fully opaque
        self.sld_alpha.setFixedWidth(SLIDER_WIDTH_PX)
        self.lbl_alpha_val = QtWidgets.QLabel("100%")
        self.lbl_alpha.setContentsMargins(12,0,6,0)     # a tiny gap from point size

        # comment rows — span ALL 13 columns
        grid.addWidget(self.lbl_comment, 0, 0, 1, 16)
        grid.addWidget(self.txt_comment, 1, 0, 1, 16)

        # buttons row
        grid.addWidget(self.btn_prev,  2, 0, 1, 2)
        grid.addWidget(self.btn_next,  2, 2, 1, 2)
        grid.addWidget(self.btn_revise, 2, 4, 1, 2)
        grid.addWidget(self.btn_png,   2, 6, 1, 1)
        grid.addWidget(self.btn_reset,  2, 7, 1, 1)
        grid.addWidget(self.btn_savec, 2, 8)

        # empty gap between buttons and slider cluster
        grid.setColumnStretch(9, 1)

        # slider cluster, right-justified
        grid.addWidget(self.lbl_ps,     2, 10, 1, 1, alignment=QtCore.Qt.AlignRight)
        grid.addWidget(self.sld_ps,     2, 11, 1, 1, alignment=QtCore.Qt.AlignRight)
        grid.addWidget(self.lbl_ps_val, 2, 12, 1, 1, alignment=QtCore.Qt.AlignLeft)
        grid.addWidget(self.lbl_alpha,     2, 13, 1, 1, alignment=QtCore.Qt.AlignRight)
        grid.addWidget(self.sld_alpha,     2, 14, 1, 1, alignment=QtCore.Qt.AlignRight)
        grid.addWidget(self.lbl_alpha_val, 2, 15, 1, 1, alignment=QtCore.Qt.AlignLeft)

        self.status=self.statusBar()
        self.pbar = _make_pbar( max(self.total, 1), initial=1, desc="Scenes", leave=False, ncols=65, 
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}" )

        # connect
        self.btn_prev.clicked.connect(lambda: self.shift(-1))
        self.btn_next.clicked.connect(lambda: self.shift(1))
        self.btn_revise.clicked.connect(self.move_to_revise)
        self.btn_png.clicked.connect(self.save_png)
        self.btn_reset.clicked.connect(self.reset_view)
        self.btn_savec.clicked.connect(self.save_comment)
        self.btn_export_xl.clicked.connect(self.export_excel)
        self.btn_folders.clicked.connect(self.edit_folders)
        self.chk_overlay.toggled.connect(self.toggle_overlay)
        self.chk_show_ann.toggled.connect(self.toggle_annotations) 
        self.chk_resume.toggled.connect(self._on_resume_toggle)
        self.sld_ps.valueChanged.connect(self.on_slider)
        self.sld_alpha.valueChanged.connect(self.on_alpha_slider)

        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self.save_png)        

        if self.total == 0:
            self.status.showMessage("No annotation files found. Click Folders to configure.", 10000)
        else:
            self.update_scene(first=True)
            
        QtWidgets.QApplication.instance().installEventFilter(self)

    # ----- data collection -----
    def collect(self) -> Tuple[List[str], Dict[str, str], Dict[str, Optional[str]]]:
        """
        Build keys from the relative path under each ANNO_DIR (without extension),
        e.g., 'Concrete/branched/branched_1'. This avoids filename collisions and
        lets us pair to the matching Original via same-index roots.
        """
        anno_map: Dict[str, str] = {}
        orig_map: Dict[str, Optional[str]] = {}

        for i, anno_root in enumerate(self.ANNO_DIRS):
            for ext in ALLOWED_EXT:
                pat = os.path.join(anno_root, f"**/*{ext}")
                for p in glob.glob(pat, recursive=True):
                    if not os.path.isfile(p):
                        continue
                    rel = os.path.relpath(p, anno_root)                 # e.g., Concrete\branched\branched_1.ply
                    stem_rel = os.path.splitext(rel)[0].replace("\\", "/")

                    # >>> NEW: key is prefixed with the anno-root index to avoid collisions
                    key = f"{i:02d}/{stem_rel}"
                    anno_map[key] = p

                    # Pair with Original of the SAME index, preserving the same rel path
                    ori = None
                    if i < len(self.ORIG_DIRS):
                        cand = os.path.join(self.ORIG_DIRS[i], rel)
                        if os.path.exists(cand):
                            ori = cand
                        else:
                            # fallback: try any allowed extension under same rel stem
                            stem_rel = os.path.splitext(rel)[0]
                            for e in ALLOWED_EXT:
                                q = os.path.join(self.ORIG_DIRS[i], stem_rel + e)
                                if os.path.exists(q):
                                    ori = q
                                    break
                    orig_map[key] = ori

        stems = sorted(anno_map.keys(), key=natural_k)
        return stems, anno_map, orig_map


    def toggle_annotations(self, checked: bool):
        self.show_annotations = bool(checked)
        self.update_scene()
        
    def _on_resume_toggle(self, checked: bool):
        self.settings["continue_where_left"] = bool(checked)
        # if turning on, immediately record current position
        if checked and self.total:
            self.settings["last_stem"] = self.stems[self.idx]
        save_settings(self.settings)

    # ----- plotting helpers (VisPy)
    def update_scene(self, *, first=False):
        if self.total==0: return
        self.canvas.clear()

        name = self.stems[self.idx]                     # key is relative path (no ext)
        anno_p = self.anno_map.get(name)
        # derive stem-only for compatibility (used for title & old comments)
        stem_only = os.path.splitext(os.path.basename(anno_p or ""))[0]
        orig_p = self.orig_map.get(name) or find_file(stem_only, self.ORIG_DIRS)

        file_name = file_stem(anno_p or orig_p or name)  # for window title/status
        key = name  # comments key (new, collision-proof)
        abs_key = os.path.abspath(anno_p) if anno_p else None

        # load point clouds
        orig = load_pc(orig_p)
        anno = load_pc(anno_p)

        # left: original
        if orig is not None and len(orig.points)>0:
            xyzL, cL = pc_to_xyz_rgb(orig)
            self.canvas.set_left(xyzL, cL, self.point_size*MARKER_SCALE_VISPY)
        else:
            self.canvas.set_left(None, None, self.point_size*MARKER_SCALE_VISPY)

        # right: overlay vs as-is
        if self.overlay_mode:
            # base = original colors
            if orig is not None and len(orig.points)>0:
                xyzB, cB = pc_to_xyz_rgb(orig)
                self.canvas.set_right_base(xyzB, cB, self.point_size*MARKER_SCALE_VISPY)
            else:
                self.canvas.set_right_base(None, None, self.point_size*MARKER_SCALE_VISPY)

            # overlay = red in annotation
            if anno is not None and len(anno.points)>0:
                xyzA, cA = pc_to_xyz_rgb(anno)
                m = red_mask(cA)
                if self.show_annotations and np.any(m):
                    self.canvas.set_right_overlay(xyzA[m], cA[m], self.point_size*MARKER_SCALE_VISPY)
                else:
                    self.canvas.set_right_overlay(None, None, self.point_size*MARKER_SCALE_VISPY)
            else:
                self.canvas.set_right_overlay(None, None, self.point_size*MARKER_SCALE_VISPY)
        else:
            # annotation as-is (split so red draws on top)
            if anno is not None and len(anno.points) > 0:
                xyzR, cR = pc_to_xyz_rgb(anno)
                m = red_mask(cR)

                # draw non-red first as base
                if np.any(~m):
                    self.canvas.set_right_base(xyzR[~m], cR[~m], self.point_size * MARKER_SCALE_VISPY)
                else:
                    self.canvas.set_right_base(None, None, self.point_size * MARKER_SCALE_VISPY)
                    
                # then draw red on top (depth_test=False inside set_right_overlay)
                if self.show_annotations and np.any(m):
                    self.canvas.set_right_overlay(xyzR[m], cR[m], self.point_size * MARKER_SCALE_VISPY)
                else:
                    self.canvas.set_right_overlay(None, None, self.point_size * MARKER_SCALE_VISPY)

            else:
                self.canvas.set_right_base(None, None, self.point_size * MARKER_SCALE_VISPY)
                self.canvas.set_right_overlay(None, None, self.point_size * MARKER_SCALE_VISPY)

        self.setWindowTitle(f"{APP_NAME} — {file_name}   ({self.idx+1}/{self.total})")

        # self.txt_comment.blockSignals(True)
        # self.txt_comment.setPlainText(self.comments.get(key, ""))
        # self.txt_comment.blockSignals(False)
        
        self.txt_comment.blockSignals(True)
        self.txt_comment.setPlainText(
            (self.comments.get(abs_key) if abs_key else "") or
            self.comments.get(key, "")
        )
        self.txt_comment.blockSignals(False)

        # advance progress only the first time this stem is viewed
        curr = self.stems[self.idx]
        if not first and curr not in self._seen:
            self.pbar.update(1)
            self._seen.add(curr)
            
        self.status.showMessage(f"Viewing: {file_name}")
        
        # fit to canvas in top view every time a new scene is loaded
        self.canvas.fit_to_data_top()
        
        # at the end of update_scene()
        self.canvas.set_titles(
            "Original",
            "Annotation (overlay)" if self.overlay_mode else "Annotation"
        )


    # ——— overlay toggle ————————————————————————————————————————
    def toggle_overlay(self, checked: bool):
        self.overlay_mode = bool(checked)
        self.settings["overlay_red_on_blue"] = self.overlay_mode
        save_settings(self.settings)
        self.canvas.set_titles("Original", "Annotation (overlay)" if self.overlay_mode else "Annotation")
        self.update_scene()

    # ----- slider (instant with VisPy)
    def on_slider(self, val: int):
        self.point_size = max(0.1, val/10.0)
        self.lbl_ps_val.setText(f"{self.point_size:.1f}")
        self.canvas.set_point_size(self.point_size*MARKER_SCALE_VISPY)
        
    def on_alpha_slider(self, val: int):
        alpha = max(0.0, min(1.0, val / 100.0))
        self.lbl_alpha_val.setText(f"{int(alpha*100)}%")
        self.canvas.set_overlay_alpha(alpha)


    def _remember_position(self):
        if self.total and self.settings.get("continue_where_left", False):
            self.settings["last_stem"] = self.stems[self.idx]
            self.settings["last_index"] = self.idx
            save_settings(self.settings)

    # ----- nav & actions
    def shift(self, d: int):
        if self.total==0: return
        self.save_comment()  # autosave on nav
        self.idx = (self.idx + d) % self.total
        self._remember_position()
        self.update_scene()

    def closeEvent(self, event):  # QCloseEvent
        self._remember_position()
        super().closeEvent(event) 

    def save_comment(self):
        if self.total == 0:
            return
        name = self.stems[self.idx]                     # existing key (index-prefixed relpath)
        anno_p = self.anno_map.get(name)
        abs_key = os.path.abspath(anno_p) if anno_p else None

        txt = self.txt_comment.toPlainText().strip()
        if not txt:
            # Do NOT delete on empty text (protects comments during revise/autosave)
            return

        # Primary: absolute annotation path (collision-proof, future-proof)
        if abs_key:
            self.comments[abs_key] = txt

        # Secondary: keep writing legacy key for backward compatibility
        self.comments[name] = txt

        save_comments(self.comments)
        self.status.showMessage(f"Comment saved for {abs_key or name} → {COMMENTS}", 2000)


    def move_to_revise(self):
        if self.total==0: return
        self.save_comment()
        name=self.stems[self.idx]
        ap=self.anno_map.get(name)
        if not ap or not os.path.exists(ap):
            self.status.showMessage(f"Annotation not found for {name}", 4000); return
        try:
            src_idx = next(i for i,root in enumerate(self.ANNO_DIRS)
                           if os.path.abspath(ap).startswith(os.path.abspath(root)))
            dst_root = self.REVISE_DIRS[src_idx]
        except StopIteration:
            self.status.showMessage(f"{ap} is not in ANNO_DIRS; skipped.", 4000); return
        ensure_dir(dst_root); dst = os.path.join(dst_root, os.path.basename(ap))
        try: shutil.move(ap, dst)
        except Exception as e:
            self.status.showMessage(f"Move failed: {e}", 6000); return
            
        # ===================== INSERT THIS BLOCK HERE =====================
        # <<< NEW: migrate comment abs-path key, and remember revised file >>>
        ap_abs  = os.path.abspath(ap)
        dst_abs = os.path.abspath(dst)

        # migrate absolute-path comment key (if present)
        migrated = False
        if ap_abs in self.comments:
            self.comments[dst_abs] = self.comments.pop(ap_abs)
            migrated = True
        if migrated:
            save_comments(self.comments)

        # remember revised file for Excel export
        if dst_abs not in self.settings["REVISED_FILES"]:
            self.settings["REVISED_FILES"].append(dst_abs)

        # pair ORIGINAL path using same src-root + relpath
        try:
            src_root = os.path.abspath(self.ANNO_DIRS[src_idx])
            rel = os.path.relpath(ap_abs, src_root)  # rel inside the ANNO root
            if src_idx < len(self.ORIG_DIRS):
                cand_orig = os.path.join(self.ORIG_DIRS[src_idx], rel)
                if os.path.exists(cand_orig):
                    self.settings["REVISED_ORIGS"][dst_abs] = os.path.abspath(cand_orig)
        except Exception:
            pass

        save_settings(self.settings)
        # =================== END OF INSERTED BLOCK ========================
    
        del self.anno_map[name]
        self.stems.pop(self.idx)
        self.total=len(self.stems)
        self.pbar.total=max(self.total,1); self.pbar.refresh()
        save_comments(self.comments)
        if self.total==0:
            QtWidgets.QMessageBox.information(self,"Done","All files reviewed — closing.")
            QtWidgets.qApp.quit()
        else:
            self.idx %= self.total
            self.update_scene()

    def save_png(self):
        if self.total==0: return
        name=self.stems[self.idx]
        out=f"{name}.png"
        try:
            rgba = self.canvas.screenshot_rgba()   # (H, W, 4) uint8
            Image.fromarray(rgba).save(out)
            self.status.showMessage(f"Saved PNG → {out}", 3000)
        except Exception as e:
            self.status.showMessage(f"Save PNG failed: {e}", 6000)

    # ----- Excel export
    def _ann_stats(self, st: str) -> Tuple[int, str]:
        p = self.anno_map.get(st)
        if not p or not os.path.exists(p):
            return 0, "none"
        try:
            pc = load_pc(p)
            if pc is None or len(pc.points) == 0:
                return 0, "none"
            _, c = pc_to_xyz_rgb(pc)
            if c.size == 0:
                return 0, "none"

            # Be robust if some files carry 0–255 colors
            if np.nanmax(c) > 1.0:
                c = np.clip(c / 255.0, 0.0, 1.0)

            cnt = int(np.count_nonzero(red_mask(c)))
            return cnt, ("crack" if cnt > 0 else "none")
        except Exception:
            return 0, "none"

    def export_excel(self):
        self.save_comment()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Excel", str(Path(self.last_export)/"comments_export.xlsx"),
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if not path: return
        if not path.lower().endswith(".xlsx"): path += ".xlsx"
        rows = []

        # Helper: count red points given an absolute annotation file path
        def _stats_from_path(anno_path: str) -> Tuple[int, str]:
            if not anno_path or not os.path.exists(anno_path):
                return 0, "none"
            try:
                pc = load_pc(anno_path)
                if pc is None or len(pc.points) == 0:
                    return 0, "none"
                _, c = pc_to_xyz_rgb(pc)
                if c.size == 0:
                    return 0, "none"
                if np.nanmax(c) > 1.0:
                    c = np.clip(c / 255.0, 0.0, 1.0)
                cnt = int(np.count_nonzero(red_mask(c)))
                return cnt, ("crack" if cnt > 0 else "none")
            except Exception:
                return 0, "none"

        # 3a) Export currently listed items (self.stems)
        for st in sorted(self.stems, key=natural_k):
            anno_p   = self.anno_map.get(st, "") or ""
            orig_p   = self.orig_map.get(st, "") or ""
            anno_abs = os.path.abspath(anno_p) if anno_p else ""
            orig_abs = os.path.abspath(orig_p) if orig_p else ""

            # comment: absolute-path key first, then legacy key
            cmt = (self.comments.get(anno_abs) or self.comments.get(st, "") or "").strip()
            has = bool(cmt)

            cnt, cls = self._ann_stats(st)   # uses anno_map; fine for in-list items

            rows.append({
                "filename":          anno_abs,   # ALWAYS absolute
                "annotation_path":   anno_abs,
                "original_path":     orig_abs,
                "comment":           cmt,
                "has_comment":       has,
                "annotations_count": cnt,
                "annotations_class": cls,
            })

        # 3b) Append revised items (moved out of ANNO_DIRS)
        revised_files = [p for p in self.settings.get("REVISED_FILES", []) if os.path.exists(p)]
        revised_origs = self.settings.get("REVISED_ORIGS", {})

        for anno_abs in revised_files:
            orig_abs = revised_origs.get(anno_abs, "")
            cmt = (self.comments.get(anno_abs) or "").strip()
            has = bool(cmt)
            cnt, cls = _stats_from_path(anno_abs)

            rows.append({
                "filename":          anno_abs,   # ALWAYS absolute
                "annotation_path":   anno_abs,
                "original_path":     orig_abs,
                "comment":           cmt,
                "has_comment":       has,
                "annotations_count": cnt,
                "annotations_class": cls,
            })

        df = pd.DataFrame(rows)

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                df.to_excel(xw, index=False, sheet_name="Comments")
            self.last_export=str(Path(path).parent)
            self.settings["last_export_path"]=self.last_export
            save_settings(self.settings)
            self.status.showMessage(f"Exported Excel → {path}", 6000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Export failed",str(e))

    # ----- folders
    def edit_folders(self, *, startup=False):
        dlg=FolderPrefsDialog(self.settings, self)
        while True:
            result = dlg.exec_()
            if result == QtWidgets.QDialog.Accepted:
                vals=dlg.values()
                self.ORIG_DIRS=vals["ORIG_DIRS"]; self.ANNO_DIRS=vals["ANNO_DIRS"]; self.REVISE_DIRS=vals["REVISE_DIRS"]
                self.settings.update(vals); save_settings(self.settings)
                self.stems, self.anno_map, self.orig_map = self.collect()
                if not self.stems:
                    QtWidgets.QMessageBox.warning(self, "No files",
                        "No annotation *.ply / *.pcd found in the selected folders. Please select again.")
                    dlg = FolderPrefsDialog(self.settings, self)
                    continue
                self.idx=0; self.total=len(self.stems)
                
                self._seen.clear()
                self.pbar.n = 0
                self.pbar.total = max(self.total, 1)
                self.pbar.refresh()

                if not startup: self.update_scene(first=True)
            break
        
    def reset_view(self):
        self.canvas.reset_view()
        
    
    def eventFilter(self, obj, ev):        
        # --- NEW: 'O' / 'o' toggles the "Overlay on Original PC" checkbox ---
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_O:
            fw = QtWidgets.QApplication.focusWidget()
            in_text = isinstance(fw, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit))

            # Toggle overlay if:
            #  - outside text inputs with plain 'o' (no modifiers), OR
            #  - Shift+O anywhere (lets you trigger even while typing by holding Shift)
            if (not in_text and ev.modifiers() == QtCore.Qt.NoModifier) or \
               (ev.modifiers() & QtCore.Qt.ShiftModifier):
                self.chk_overlay.toggle()
                return True  # consume event

        # (existing Reset View 'r/R' handler below remains unchanged)
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_R:
            fw = QtWidgets.QApplication.focusWidget()
            in_text = isinstance(fw, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit))
            if (not in_text and ev.modifiers() == QtCore.Qt.NoModifier) or \
               (ev.modifiers() & QtCore.Qt.ShiftModifier):
                self.reset_view()
                return True
            
        # --- NEW: 'A' / 'a' toggles red annotations visibility ---
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_A:
            fw = QtWidgets.QApplication.focusWidget()
            in_text = isinstance(fw, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit))

            # Toggle when:
            #  - outside text inputs with plain 'a' (no modifiers), OR
            #  - Shift+A anywhere (lets you toggle even while typing)
            if (not in_text and ev.modifiers() == QtCore.Qt.NoModifier) or \
               (ev.modifiers() & QtCore.Qt.ShiftModifier):
                self.chk_show_ann.toggle()
                return True
        
        # --- NEW: 'M' / 'm' revises (moves) current file to REVISE_DIRS ---
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_M:
            fw = QtWidgets.QApplication.focusWidget()
            in_text = isinstance(fw, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit))

            # Toggle when:
            #  - outside text inputs with plain 'm' (no modifiers), OR
            #  - Shift+M anywhere (lets you toggle even while typing)
            if (not in_text and ev.modifiers() == QtCore.Qt.NoModifier) or \
               (ev.modifiers() & QtCore.Qt.ShiftModifier):
                self.move_to_revise()
                return True
            
        
        # --- NEW: ->  Left arrow to move to next point cloud ---
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_Left:
            fw = QtWidgets.QApplication.focusWidget()
            in_text = isinstance(fw, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit))

            # Toggle when:
            #  - outside text inputs with plain '->' (no modifiers), OR
            #  - Shift+-> anywhere (lets you toggle even while typing)
            if (not in_text and ev.modifiers() == QtCore.Qt.NoModifier) or \
               (ev.modifiers() & QtCore.Qt.ShiftModifier):
                self.shift(-1)
                return True
            
            
        # --- NEW: <-  Right arrow to move to next point cloud ---
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_Right:
            fw = QtWidgets.QApplication.focusWidget()
            in_text = isinstance(fw, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit, QtWidgets.QLineEdit))

            # Toggle when:
            #  - outside text inputs with plain '<-' (no modifiers), OR
            #  - Shift+<- anywhere (lets you toggle even while typing)
            if (not in_text and ev.modifiers() == QtCore.Qt.NoModifier) or \
               (ev.modifiers() & QtCore.Qt.ShiftModifier):
                self.shift(1)
                return True

        return super().eventFilter(obj, ev)


def main():
    t0=time.time()
    settings=load_settings()
    app=QtWidgets.QApplication(sys.argv)
    win=ReviewerApp(settings)
    win.showMaximized()
    ret=app.exec_()
    _safe_print(f"\nFinished in {time.time()-t0:0.1f} s")
    _safe_print("Comments JSON :", COMMENTS)
    _safe_print("Settings JSON :", SETTINGS)
    sys.exit(ret)

if __name__=="__main__":
    main()