#!/usr/bin/env python3
"""
Blind Eye — CSI Motion Sensing Dashboard
PyQt6 desktop app per MacOS
"""

import sys
import threading
import re
import time
import csv
import json
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QFrame, QSplitter, QStatusBar, QFileDialog, QDialog, QSlider,
    QDoubleSpinBox, QCheckBox, QLineEdit, QDialogButtonBox, QGroupBox,
    QSpinBox, QMessageBox, QListWidget, QInputDialog, QFormLayout,
    QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette, QCursor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("/Users/federico/esp-csi/examples/get-started/csi_recv/data")
_MODEL_DIR    = Path(__file__).parent
_SETTINGS_FILE = _MODEL_DIR / 'settings.json'
_PROFILES_DIR  = DATA_DIR / 'profiles'

# ── Settings ─────────────────────────────────────────────────────────────────
_SETTINGS_DEFAULTS = {
    'port':           '/dev/cu.usbserial-110',
    'baud':           115200,
    'tx_mac':         '1a:00:00:00:00:00',
    'filter_mac':     True,
    'room_w':         5.7,
    'room_h':         3.4,
    'tx_pos':         [1.0, 2.8],
    'rx_pos':         [4.7, 0.6],
    'conf_threshold': 0.60,
    'notifications':  True,
    'notif_cooldown': 10,
}

def _load_settings():
    s = dict(_SETTINGS_DEFAULTS)
    try:
        with open(_SETTINGS_FILE) as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s

def _save_settings_to_file(d):
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass

_CFG = _load_settings()

# ── Configurazione ─────────────────────────────────────────────────────────
PORT       = _CFG['port']
BAUD       = _CFG['baud']
HISTORY    = 200
TX_MAC     = _CFG['tx_mac']
FILTER_MAC = _CFG['filter_mac']
ROOM_W     = _CFG['room_w']
ROOM_H     = _CFG['room_h']
TX_POS     = tuple(_CFG['tx_pos'])
RX_POS     = tuple(_CFG['rx_pos'])
ML_CONF_THRESHOLD = float(_CFG.get('conf_threshold', 0.60))

# ── Modello ML ───────────────────────────────────────────────────────────────
def _load_model():
    try:
        import joblib
        m = joblib.load(_MODEL_DIR / 'csi_model.pkl')
        s = joblib.load(_MODEL_DIR / 'csi_scaler.pkl')
        f = json.load(open(_MODEL_DIR / 'csi_features.json'))
        return m, s, f
    except Exception:
        return None, None, None

ML_MODEL, ML_SCALER, ML_FEATURES = _load_model()
ML_LABELS = {
    'empty':       ('STANZA VUOTA',       '#00ff88'),
    'motion':      ('MOVIMENTO',          '#ffaa00'),
    'static':      ('PRESENZA STATICA',   '#ff4466'),
    'semi_static': ('PRESENZA SEMI-FERMA','#aa44ff'),
}

# ── Colori ──────────────────────────────────────────────────────────────────
BG_DARK      = "#0d0d14"
BG_PANEL     = "#13131e"
BG_CARD      = "#1a1a2e"
ACCENT_BLUE  = "#00cfff"
ACCENT_GREEN = "#00ff88"
ACCENT_RED   = "#ff4466"
ACCENT_AMBER = "#ffaa00"
TEXT_PRIMARY = "#e8e8f0"
TEXT_MUTED   = "#6b6b8a"
BORDER       = "#2a2a3e"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'SF Mono', 'Menlo', monospace;
    font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG_PANEL};
    border-radius: 8px;
}}
QTabBar::tab {{
    background: {BG_CARD};
    color: {TEXT_MUTED};
    padding: 10px 24px;
    border: none;
    font-size: 11px;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    color: {ACCENT_BLUE};
    border-bottom: 2px solid {ACCENT_BLUE};
    background: {BG_PANEL};
}}
QPushButton {{
    background: transparent;
    color: {ACCENT_BLUE};
    border: 1px solid {ACCENT_BLUE};
    padding: 8px 20px;
    border-radius: 4px;
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton:hover {{ background: rgba(0,207,255,0.1); }}
QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {TEXT_MUTED}; }}
QPushButton#danger {{ color: {ACCENT_RED}; border-color: {ACCENT_RED}; }}
QPushButton#success {{ color: {ACCENT_GREEN}; border-color: {ACCENT_GREEN}; }}
QLabel {{ color: {TEXT_PRIMARY}; }}
QLabel#muted {{ color: {TEXT_MUTED}; font-size: 10px; letter-spacing: 1px; }}
QLabel#value {{ color: {ACCENT_BLUE}; font-size: 22px; font-weight: bold; }}
QFrame#card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QTableWidget {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
    gridline-color: {BORDER};
    font-size: 10px;
}}
QTableWidget::item {{ padding: 4px 8px; color: {TEXT_PRIMARY}; }}
QTableWidget::item:selected {{ background: rgba(0,207,255,0.15); }}
QHeaderView::section {{
    background: {BG_PANEL};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-size: 10px;
}}
QStatusBar {{
    background: {BG_PANEL};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
    font-size: 10px;
}}
QSplitter::handle {{ background: {BORDER}; }}
"""

# ── Classificatore real-time ─────────────────────────────────────────────────
_ml_window = deque(maxlen=5)   # ultimi 5 frame per majority vote

def classify_packet(parsed, delta):
    """Ritorna (label, confidenza) oppure (None, 0) se il modello non è caricato."""
    if ML_MODEL is None:
        return None, 0.0
    try:
        amp   = parsed['amp']
        phase = parsed['phase']
        n     = min(len(amp), 64)
        amp_p   = list(amp[:n])   + [0.0] * (64 - n)
        phase_p = list(phase[:n]) + [0.0] * (64 - n)
        row = amp_p + phase_p + [delta, float(parsed['rssi'])]
        x   = ML_SCALER.transform([row])
        proba = ML_MODEL.predict_proba(x)[0]
        idx   = int(np.argmax(proba))
        _ml_window.append(idx)
        # Majority vote sulla finestra
        vote = int(np.bincount(list(_ml_window),
                               minlength=len(ML_MODEL.classes_)).argmax())
        return ML_MODEL.classes_[vote], float(proba[idx])
    except Exception:
        return None, 0.0

# ── Helper UI ────────────────────────────────────────────────────────────────
class InfoIcon(QLabel):
    """Icona ⓘ che mostra un pannello informativo istantaneo al passaggio del mouse."""

    _POPUP_STYLE = f"""
        background-color: #12121f;
        color: #e8e8f0;
        border: 1px solid {ACCENT_BLUE};
        border-top: 2px solid {ACCENT_BLUE};
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'SF Mono', 'Menlo', monospace;
        font-size: 11px;
    """

    def __init__(self, tooltip_text, parent=None):
        super().__init__("ⓘ", parent)
        self._tip_text = tooltip_text
        self._popup    = None
        self.setFixedSize(18, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px;"
            f" border: 1px solid {BORDER}; border-radius: 4px;"
            f" padding: 0px; background: transparent;"
        )
        self.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))

    def enterEvent(self, event):
        popup = QLabel()
        popup.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        popup.setStyleSheet(self._POPUP_STYLE)
        popup.setTextFormat(Qt.TextFormat.RichText)
        popup.setText(self._tip_text)
        popup.setWordWrap(True)
        popup.setMaximumWidth(400)
        popup.adjustSize()
        # Posiziona a destra e leggermente sotto il cursore
        pos = QCursor.pos()
        popup.move(pos.x() + 14, pos.y() + 14)
        popup.show()
        self._popup = popup
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._popup:
            self._popup.close()
            self._popup = None
        super().leaveEvent(event)


def info_row(title, tooltip_text):
    """Intestazione grafico con titolo + icona ⓘ con pannello info istantaneo."""
    w = QWidget()
    w.setFixedHeight(22)
    w.setStyleSheet("background: transparent;")
    hl = QHBoxLayout(w)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(6)
    lbl = QLabel(title)
    lbl.setObjectName("muted")
    hl.addWidget(lbl)
    hl.addWidget(InfoIcon(tooltip_text))
    hl.addStretch()
    return w

# ── Parser CSI ──────────────────────────────────────────────────────────────
CSI_RE = re.compile(r'CSI_DATA,(-?\d+),([^,]+),(-?\d+),(\d+),(\d+),(\d+),(\d+).*?\[([^\]]+)\]')

def parse_csi_line(line):
    m = CSI_RE.search(line)
    if not m:
        return None
    try:
        mac  = m.group(2)
        rssi = int(m.group(3))
        raw  = list(map(int, m.group(8).split(',')))
        if len(raw) < 4:
            return None
        pairs = [(raw[i], raw[i+1]) for i in range(0, len(raw)-1, 2)]
        amp   = np.array([np.sqrt(r**2 + im**2) for im, r in pairs])
        phase = np.array([np.arctan2(im, r)      for im, r in pairs])
        return {'mac': mac, 'rssi': rssi, 'amp': amp, 'phase': phase, 'raw': raw}
    except Exception:
        return None

# ── Serial worker ────────────────────────────────────────────────────────────
def _find_port():
    """Preferisce PORT se disponibile, altrimenti prende la porta con il numero più lungo (RX)."""
    import glob as _glob
    candidates = _glob.glob('/dev/cu.usbserial-*')
    if not candidates:
        return PORT
    if PORT in candidates:
        return PORT
    # usbserial-110 ha nome più lungo di usbserial-10: prende quella più lunga
    return max(candidates, key=len)

class SerialWorker(QObject):
    data_received  = pyqtSignal(dict)
    status_changed = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

    def _run(self):
        import serial
        import serial.serialutil
        while self._running:
            port = _find_port()
            try:
                ser = serial.Serial(port, BAUD, timeout=1)
                self.status_changed.emit(f"Connesso a {port}", "ok")
                while self._running:
                    try:
                        raw = ser.readline().decode('utf-8', errors='ignore').strip()
                        if raw:
                            parsed = parse_csi_line(raw)
                            if parsed:
                                self.data_received.emit(parsed)
                    except serial.serialutil.SerialException:
                        break  # porta scomparsa → riconnetti
                    except Exception:
                        time.sleep(0.01)
                ser.close()
            except Exception as e:
                self.status_changed.emit(f"In attesa di dispositivo… ({e})", "error")
            if self._running:
                time.sleep(2)  # pausa prima di ritentare

# ── Canvas ───────────────────────────────────────────────────────────────────
class DarkCanvas(FigureCanvas):
    def __init__(self, figsize=(6, 3)):
        self.fig = Figure(figsize=figsize, facecolor=BG_CARD)
        super().__init__(self.fig)
        self.setStyleSheet(f"background-color: {BG_CARD};")

def style_ax(ax):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.5)

# ── Suoni ────────────────────────────────────────────────────────────────────
def play_sound(name, repeat=1, gap=0.15):
    path = f'/System/Library/Sounds/{name}.aiff'
    try:
        for i in range(repeat):
            subprocess.Popen(['afplay', path])
            if i < repeat - 1:
                time.sleep(gap)
    except Exception:
        QApplication.beep()

# ── Notifiche macOS ──────────────────────────────────────────────────────────
_last_notif_time  = 0.0
_last_notif_label = None

def notify_mac(title, subtitle=''):
    global _last_notif_time, _last_notif_label
    if not _CFG.get('notifications', True):
        return
    now = time.time()
    cooldown = float(_CFG.get('notif_cooldown', 10))
    key = f"{title}|{subtitle}"
    if now - _last_notif_time < cooldown and key == _last_notif_label:
        return
    _last_notif_time  = now
    _last_notif_label = key
    script = (
        f'display notification "{subtitle}" '
        f'with title "Blind Eye" subtitle "{title}" sound name "default"'
    )
    try:
        subprocess.Popen(['osascript', '-e', script])
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — MONITOR LIVE
# ══════════════════════════════════════════════════════════════════════════
class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        self.csi_history    = deque(maxlen=HISTORY)
        self.rssi_history   = deque(maxlen=HISTORY)
        self.motion_history = deque(maxlen=HISTORY)

        # Calibrazione
        self.baseline_mean = None
        self.baseline_std  = None
        self.calib_phase   = None
        self.calib_counter = 0
        self.calib_buffer  = []

        # Isteresi
        self.current_state   = 0
        self.candidate_state = 0
        self.state_counter   = 0
        self.CONFIRM_UP      = 4
        self.CONFIRM_DOWN    = 20

        self.packet_count = 0
        self.fps_count    = 0
        self.last_time    = time.time()

        self._build_ui()
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self._calib_tick)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Metriche
        metrics = QHBoxLayout()
        self.lbl_rssi    = self._metric("RSSI",      "— dBm", metrics)
        self.lbl_packets = self._metric("PACCHETTI", "0",     metrics)
        self.lbl_delta   = self._metric("DELTA CSI", "—",     metrics)
        self.lbl_fps     = self._metric("FPS",       "—",     metrics)
        layout.addLayout(metrics)

        # Grafici
        graphs = QHBoxLayout()
        graphs.setSpacing(12)

        amp_card = self._card()
        al = QVBoxLayout(amp_card); al.setContentsMargins(12,12,12,12)
        al.addWidget(info_row("AMPIEZZA SOTTOPORTANTI",
            "Ampiezza del segnale per ciascuna delle <b>64 sottoportanti Wi-Fi</b>"
            " nel pacchetto CSI più recente.<br><br>"
            "<b>Asse X</b> — indice sottoportante (0–63)<br>"
            "<b>Asse Y</b> — ampiezza (modulo del numero complesso <b>IQ</b>,"
            " unità adimensionale)<br><br>"
            "<u>Come si legge:</u> il profilo varia lentamente a stanza vuota."
            " Quando una persona si muove le <b>sottoportanti centrali</b>"
            " mostrano oscillazioni più marcate rispetto a quelle ai bordi."))
        self.canvas_amp = DarkCanvas((5, 2.5))
        self.ax_amp = self.canvas_amp.fig.add_subplot(111)
        self.canvas_amp.fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.13)
        style_ax(self.ax_amp)
        al.addWidget(self.canvas_amp)
        graphs.addWidget(amp_card)

        var_card = self._card()
        vl = QVBoxLayout(var_card); vl.setContentsMargins(12,12,12,12)
        vl.addWidget(info_row("ATTIVITÀ NEL TEMPO",
            "Andamento temporale del <b>delta CSI</b>: quanto il canale radio"
            " si è discostato dalla media delle ultime 10 misure.<br><br>"
            "<b>Asse X</b> — frame ricevuti (ultimi 200)<br>"
            "<b>Asse Y</b> — delta CSI (adimensionale, valori tipici 0–5)<br>"
            "<b>Linea gialla</b> tratteggiata — soglia movimento lieve<br>"
            "<b>Linea rossa</b> tratteggiata — soglia movimento umano<br><br>"
            "<u>Come si legge:</u> baseline bassa = stanza vuota."
            " <u>Un picco sostenuto sopra la soglia rossa</u>"
            " indica presenza umana in movimento."))
        self.canvas_var = DarkCanvas((5, 2.5))
        self.ax_var = self.canvas_var.fig.add_subplot(111)
        self.canvas_var.fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.13)
        style_ax(self.ax_var)
        vl.addWidget(self.canvas_var)
        graphs.addWidget(var_card)

        layout.addLayout(graphs)

        # Pannello stato
        status_card = self._card()
        sl = QHBoxLayout(status_card)
        sl.setContentsMargins(20, 14, 20, 14)
        sl.setSpacing(16)

        # LED
        self.led = QLabel("●")
        self.led.setFont(QFont("monospace", 24))
        self.led.setStyleSheet(f"color: {TEXT_MUTED};")
        self.led.setFixedWidth(30)
        sl.addWidget(self.led)

        # Testo stato + dettaglio
        txt = QVBoxLayout(); txt.setSpacing(3)
        self.lbl_stato = QLabel("IN ATTESA DI DATI...")
        self.lbl_stato.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 15px; letter-spacing: 2px; font-weight: bold;")
        self.lbl_dettaglio = QLabel("")
        self.lbl_dettaglio.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        txt.addWidget(self.lbl_stato)
        txt.addWidget(self.lbl_dettaglio)
        sl.addLayout(txt, 1)

        # Badge modalità — riquadro separato
        self.badge_card = QFrame()
        self.badge_card.setObjectName("card")
        self.badge_card.setFixedSize(90, 48)
        bl = QVBoxLayout(self.badge_card)
        bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(2)
        lbl_badge_title = QLabel("MODALITÀ")
        lbl_badge_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_badge_title.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 8px; letter-spacing: 1px;")
        self.lbl_mode_badge = QLabel("EURISTICA")
        self.lbl_mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mode_badge.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        bl.addWidget(lbl_badge_title)
        bl.addWidget(self.lbl_mode_badge)
        sl.addWidget(self.badge_card)

        # Countdown calibrazione (nascosto di default)
        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setStyleSheet(
            f"color: {ACCENT_AMBER}; font-size: 36px; font-weight: bold;")
        self.lbl_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_countdown.setFixedWidth(56)
        self.lbl_countdown.setVisible(False)
        sl.addWidget(self.lbl_countdown)

        self.btn_cal = QPushButton("⊙  CALIBRA BASELINE")
        self.btn_cal.setObjectName("success")
        self.btn_cal.clicked.connect(self._start_calibration)
        sl.addWidget(self.btn_cal)

        layout.addWidget(status_card)

    def _card(self):
        f = QFrame(); f.setObjectName("card"); return f

    def _lbl_muted(self, text):
        l = QLabel(text); l.setObjectName("muted"); return l

    def _metric(self, label, value, parent):
        f = self._card(); f.setFixedHeight(80)
        vl = QVBoxLayout(f); vl.setContentsMargins(16,12,16,12)
        ll = QLabel(label); ll.setObjectName("muted")
        lv = QLabel(value); lv.setObjectName("value")
        lv.setFont(QFont("SF Mono", 18, QFont.Weight.Bold))
        vl.addWidget(ll); vl.addWidget(lv)
        parent.addWidget(f)
        return lv

    def update_data(self, parsed):
        """Aggiorna solo lo stato — il rendering avviene in render()."""
        amp  = parsed['amp']
        rssi = parsed['rssi']
        self.csi_history.append(amp)
        self.rssi_history.append(rssi)
        self.packet_count += 1
        self.fps_count    += 1

        now = time.time()
        if now - self.last_time >= 1.0:
            self.lbl_fps.setText(str(self.fps_count))
            self.fps_count = 0
            self.last_time = now

        self.lbl_rssi.setText(f"{rssi} dBm")
        self.lbl_packets.setText(str(self.packet_count))

        if len(self.csi_history) < 10:
            return

        min_len     = min(len(x) for x in self.csi_history)
        data        = np.array([x[:min_len] for x in self.csi_history])
        recent_mean = np.mean(data[-10:], axis=0)
        delta       = float(np.mean(np.abs(amp[:min_len] - recent_mean)))

        # Filtro outlier: scarta picchi anomali (>3× mediana recente)
        if len(self.motion_history) >= 10:
            recent = list(self.motion_history)[-20:]
            median = float(np.median(recent))
            if delta > max(median * 4.0, 0.5) and median < 3.0:
                return   # pacchetto anomalo — ignora senza aggiornare storia

        self.motion_history.append(delta)
        self.lbl_delta.setText(f"{delta:.2f}")

        if self.calib_phase == 'recording':
            self.calib_buffer.append(delta)

        if self.baseline_mean is not None:
            tg = self.baseline_mean + 0.4 * self.baseline_std
            ty = self.baseline_mean + 0.9 * self.baseline_std
        else:
            tg, ty = 1.2, 2.2

        if self.calib_phase is None and ML_MODEL is None:
            self._update_state(delta, tg, ty)

        # Salva snapshot per render()
        self._render_state = (amp, data, tg, ty)

    def render(self):
        """Chiamato dal timer centrale — ridisegna solo se ci sono dati nuovi."""
        if not hasattr(self, '_render_state') or self._render_state is None:
            return
        amp, data, tg, ty = self._render_state
        self._render_state = None
        self._draw_graphs(amp, data, tg, ty)

    def update_ml(self, name, color, conf, delta):
        """Aggiorna il pannello stato con il risultato del classificatore ML."""
        self.led.setStyleSheet(f"color: {color};")
        self.lbl_stato.setStyleSheet(
            f"color: {color}; font-size: 15px; letter-spacing: 2px;")
        self.lbl_stato.setText(name)
        thresh_pct = int(ML_CONF_THRESHOLD * 100)
        self.lbl_dettaglio.setText(
            f"conf {conf*100:.0f}%  ·  soglia {thresh_pct}%  ·  delta {delta:.2f}")
        self.lbl_mode_badge.setText("ML")
        self.lbl_mode_badge.setStyleSheet(
            f"color: {ACCENT_BLUE}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")

    def _update_state(self, delta, tg, ty):
        cand = 0 if delta < tg else (1 if delta < ty else 2)

        if cand == self.candidate_state:
            self.state_counter += 1
        else:
            self.candidate_state = cand
            self.state_counter   = 1

        if cand > self.current_state:
            if self.state_counter >= self.CONFIRM_UP:
                self.current_state = cand
        elif cand < self.current_state:
            if self.state_counter >= self.CONFIRM_DOWN:
                self.current_state = cand

        if self.current_state == 0:
            c, s, d = ACCENT_GREEN, "NESSUN MOVIMENTO",   "Ambiente stabile"
        elif self.current_state == 1:
            c, s, d = ACCENT_AMBER, "MOVIMENTO RILEVATO", "Disturbo nel canale"
        else:
            c, s, d = ACCENT_RED,   "⚠  MOVIMENTO UMANO", "Alta perturbazione del canale"

        self.led.setStyleSheet(f"color: {c};")
        self.lbl_stato.setStyleSheet(f"color: {c}; font-size: 15px; letter-spacing: 2px;")
        self.lbl_stato.setText(s)
        self.lbl_dettaglio.setText(d)
        self.lbl_mode_badge.setText("EURISTICA")
        self.lbl_mode_badge.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")

    def _draw_graphs(self, amp, data, tg=None, ty=None):
        self.ax_amp.cla(); style_ax(self.ax_amp)
        x = np.arange(len(amp))
        self.ax_amp.fill_between(x, amp, alpha=0.4, color=ACCENT_BLUE)
        self.ax_amp.plot(x, amp, color=ACCENT_BLUE, linewidth=1)
        self.ax_amp.set_xlim(0, len(amp))
        self.canvas_amp.draw_idle()

        self.ax_var.cla(); style_ax(self.ax_var)
        if len(self.motion_history) > 1:
            y = list(self.motion_history)
            self.ax_var.fill_between(range(len(y)), y, alpha=0.3, color=ACCENT_GREEN)
            self.ax_var.plot(y, color=ACCENT_GREEN, linewidth=1.2)
            if tg is not None:
                self.ax_var.axhline(tg, color=ACCENT_AMBER, linewidth=0.8, linestyle='--', alpha=0.7)
                self.ax_var.axhline(ty, color=ACCENT_RED,   linewidth=0.8, linestyle='--', alpha=0.7)
        self.canvas_var.draw_idle()

    def _start_calibration(self):
        self.calib_phase   = 'countdown'
        self.calib_counter = 10
        self.calib_buffer  = []
        self.btn_cal.setEnabled(False)
        self.lbl_countdown.setVisible(True)
        self.lbl_countdown.setText("10")
        self.lbl_stato.setText("ESCI DALLA STANZA!")
        self.lbl_stato.setStyleSheet(f"color: {ACCENT_AMBER}; font-size: 15px; letter-spacing: 2px;")
        self.led.setStyleSheet(f"color: {ACCENT_AMBER};")
        self.lbl_dettaglio.setText("Hai 10 secondi per uscire...")
        self.calib_timer.start(1000)

    def _calib_tick(self):
        if self.calib_phase == 'countdown':
            self.calib_counter -= 1
            if self.calib_counter > 0:
                self.lbl_countdown.setText(str(self.calib_counter))
            else:
                self.calib_phase   = 'recording'
                self.calib_counter = 30
                self.lbl_countdown.setText("30")
                self.lbl_stato.setText("CALIBRAZIONE IN CORSO...")
                self.lbl_stato.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 15px; letter-spacing: 2px;")
                self.led.setStyleSheet(f"color: {ACCENT_BLUE};")
                self.lbl_dettaglio.setText("Stanza vuota — non entrare")
                threading.Thread(target=lambda: play_sound('Tink', 3, 0.18), daemon=True).start()

        elif self.calib_phase == 'recording':
            self.calib_counter -= 1
            self.lbl_countdown.setText(str(self.calib_counter))
            if self.calib_counter <= 0:
                self._finish_calibration()

    def _finish_calibration(self):
        self.calib_timer.stop()
        self.calib_phase = None
        self.lbl_countdown.setVisible(False)
        self.lbl_countdown.setText("")
        self.btn_cal.setEnabled(True)

        if len(self.calib_buffer) > 10:
            self.baseline_mean = float(np.mean(self.calib_buffer))
            self.baseline_std  = max(float(np.std(self.calib_buffer)), 0.01)
            self.lbl_stato.setText("✓  BASELINE SALVATA")
            self.lbl_dettaglio.setText(
                f"Media: {self.baseline_mean:.3f}   Std: {self.baseline_std:.3f}")
            self.lbl_stato.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 15px; letter-spacing: 2px;")
            self.led.setStyleSheet(f"color: {ACCENT_GREEN};")
            self.current_state = self.candidate_state = 0
            self.state_counter = 0
        else:
            self.lbl_stato.setText("ERRORE: pochi dati ricevuti")
            self.lbl_stato.setStyleSheet(f"color: {ACCENT_RED}; font-size: 15px; letter-spacing: 2px;")

        threading.Thread(target=lambda: play_sound('Glass', 2, 0.4), daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — MAPPA RADIO
# ══════════════════════════════════════════════════════════════════════════
class TopViewTab(QWidget):
    # Griglia 57×34 pixel → ROOM_W × ROOM_H metri
    _GX = np.linspace(0, ROOM_W, 57)
    _GY = np.linspace(0, ROOM_H, 34)

    def __init__(self):
        super().__init__()
        self.heatmap       = np.zeros((34, 57))
        self.weight_map    = self._build_weights()
        self.path_grid     = self._build_path_grid()
        self.interior_mask = self._build_interior_mask()
        self.d_direct      = float(np.sqrt(
            (TX_POS[0]-RX_POS[0])**2 + (TX_POS[1]-RX_POS[1])**2))
        self._current_ellipse = None
        self._build_ui()

    def _build_weights(self):
        xx, yy = np.meshgrid(self._GX, self._GY)
        d_tx = np.sqrt((xx-TX_POS[0])**2 + (yy-TX_POS[1])**2) + 0.1
        d_rx = np.sqrt((xx-RX_POS[0])**2 + (yy-RX_POS[1])**2) + 0.1
        w = 1.0 / (d_tx * d_rx)
        return w / w.max()

    def _build_path_grid(self):
        """Per ogni pixel: distanza totale TX→punto→RX."""
        xx, yy = np.meshgrid(self._GX, self._GY)
        d_tx = np.sqrt((xx-TX_POS[0])**2 + (yy-TX_POS[1])**2)
        d_rx = np.sqrt((xx-RX_POS[0])**2 + (yy-RX_POS[1])**2)
        return d_tx + d_rx

    def _build_interior_mask(self):
        """Maschera che azzera gradualmente i contributi vicino ai muri.
        Falloff quadratico su 0.7 m — il centro pesa 1, i bordi 0."""
        margin = 0.7
        xx, yy = np.meshgrid(self._GX, self._GY)
        dist_wall = np.minimum(
            np.minimum(xx, ROOM_W - xx),
            np.minimum(yy, ROOM_H - yy)
        )
        mask = np.clip(dist_wall / margin, 0.0, 1.0) ** 2
        return mask

    def _ellipse_points(self, d_total):
        """Calcola i punti (xe, ye) dell'ellisse con d_total come somma dei raggi."""
        a = d_total / 2.0
        c = self.d_direct / 2.0
        if a <= c:
            return None, None
        b     = np.sqrt(max(a**2 - c**2, 0.0))
        angle = np.arctan2(RX_POS[1]-TX_POS[1], RX_POS[0]-TX_POS[0])
        cx    = (TX_POS[0] + RX_POS[0]) / 2.0
        cy    = (TX_POS[1] + RX_POS[1]) / 2.0
        theta = np.linspace(0, 2*np.pi, 300)
        xe_loc = a * np.cos(theta)
        ye_loc = b * np.sin(theta)
        xe = cx + xe_loc * np.cos(angle) - ye_loc * np.sin(angle)
        ye = cy + xe_loc * np.sin(angle) + ye_loc * np.cos(angle)
        return xe, ye

    def _build_ui(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(16,16,16,16)
        hdr = QHBoxLayout()
        hdr.addWidget(info_row("MAPPA RADIO — TOP VIEW",
            "Vista dall'alto della stanza con <b>heatmap di attività radio</b>."
            " Mostra dove il segnale è più perturbato in base alla geometria"
            " <b>TX–RX</b> e all'intensità del delta CSI.<br><br>"
            "<b>Asse X</b> — larghezza stanza (metri, 0–5.7 m)<br>"
            "<b>Asse Y</b> — profondità stanza (metri, 0–3.4 m)<br>"
            "<b>Colori</b> — dal viola (bassa attività) al giallo (alta attività)<br><br>"
            "<u>Come si legge:</u> le <u>zone calde</u> indicano dove il multipath"
            " è più sensibile. Con una sola coppia TX–RX la mappa è"
            " un'approssimazione ellissoidale, <b>non una localizzazione precisa</b>."))
        hdr.addStretch()
        btn = QPushButton("RESET MAPPA")
        btn.clicked.connect(self._reset)
        hdr.addWidget(btn)
        layout.addLayout(hdr)

        self.canvas = DarkCanvas((8, 5))
        self.ax = self.canvas.fig.add_subplot(111)
        self.canvas.fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.08)
        self._draw()
        layout.addWidget(self.canvas)

        leg = QHBoxLayout()
        for color, label in [
            (ACCENT_GREEN, "TX"),
            (ACCENT_BLUE,  "RX"),
            (ACCENT_AMBER, "Calore"),
            ("#ffffff",    "Ellisse τ corrente"),
        ]:
            d = QLabel(f"● {label}")
            d.setStyleSheet(f"color: {color}; font-size: 11px;")
            leg.addWidget(d)
        leg.addStretch()
        layout.addLayout(leg)

    def _draw(self):
        ax = self.ax; ax.cla()
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.imshow(self.heatmap, extent=[0,ROOM_W,0,ROOM_H], origin='lower',
                  cmap='plasma', vmin=0, vmax=max(self.heatmap.max(), 0.01),
                  aspect='auto', interpolation='bicubic', alpha=0.85)
        ax.add_patch(plt.Rectangle((0,0),ROOM_W,ROOM_H,
                     fill=False, edgecolor=BORDER, linewidth=1.5))
        # Ellisse corrente
        if self._current_ellipse is not None:
            xe, ye = self._current_ellipse
            ax.plot(xe, ye, '-', color='white', linewidth=2.5,
                    alpha=0.9, zorder=4)
        ax.plot(*TX_POS,'s',color=ACCENT_GREEN,markersize=14,zorder=5)
        ax.annotate('TX',TX_POS,xytext=(8,4),textcoords='offset points',
                    color=ACCENT_GREEN,fontsize=9,fontweight='bold')
        ax.plot(*RX_POS,'D',color=ACCENT_BLUE,markersize=14,zorder=5)
        ax.annotate('RX',RX_POS,xytext=(8,4),textcoords='offset points',
                    color=ACCENT_BLUE,fontsize=9,fontweight='bold')
        ax.plot([TX_POS[0],RX_POS[0]],[TX_POS[1],RX_POS[1]],
                '--',color=BORDER,linewidth=0.8,alpha=0.5)
        ax.set_xlim(-0.2,ROOM_W+0.2); ax.set_ylim(-0.2,ROOM_H+0.2)
        ax.set_xlabel('m',color=TEXT_MUTED,fontsize=9)
        ax.set_ylabel('m',color=TEXT_MUTED,fontsize=9)
        self.canvas.draw_idle()

    def _reset(self):
        self.heatmap = np.zeros((34, 57))
        self._current_ellipse = None
        self._draw()

    def update_data(self, parsed, activity, tof_ns=None):
        self.heatmap *= 0.97
        # Contributo ampiezza — pesato dalla maschera interna
        self.heatmap += activity * self.weight_map * self.interior_mask * 0.1

        if tof_ns is not None:
            # τ ESP32 dominato da CFO: mappato in percorso extra 0.2–3.5 m
            d_extra = float(np.clip(abs(tof_ns) * 0.02, 0.2, 3.5))
            d_total = self.d_direct + d_extra
            sigma   = 0.3
            stamp   = np.exp(-0.5 * ((self.path_grid - d_total) / sigma) ** 2)
            # Gaussian stamp anch'esso pesato — sopprime bordi
            self.heatmap += activity * stamp * self.interior_mask * 0.4
            xe, ye = self._ellipse_points(d_total)
            self._current_ellipse = (xe, ye) if xe is not None else None

        self._draw()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — DATI & TRAINING
# ══════════════════════════════════════════════════════════════════════════
class DataTab(QWidget):
    """Unified tab: guided training sequence (top) + free recording (bottom)."""

    PHASES = [
        ('empty',  '① STANZA VUOTA',
         'Esci dalla stanza e non rientrare fino al segnale sonoro.',
         120),
        ('motion', '② CAMMINA TX → RX',
         'Cammina avanti e indietro tra le due schede in modo naturale.\n'
         'Varia velocità e traiettoria: diagonale, laterale, lento, veloce.',
         120),
        ('static', '③ PRESENZA STATICA',
         'Spostati in un punto della stanza e rimani fermo in piedi.\n'
         'Ogni ~30 secondi cambia POSIZIONE nella stanza (vicino al TX,\n'
         'al centro, vicino al RX, ai lati) — non basta cambiare postura.',
         120),
    ]
    PREP_TIME = 15

    def __init__(self):
        super().__init__()
        # Training state
        self.phase_idx     = -1
        self.phase_state   = None   # 'prep' | 'record'
        self.time_left     = 0
        self.current_label = ''
        self.buffer        = []
        self.all_rows      = []
        self.pkt_count     = 0
        self.phase_counts  = {'empty': 0, 'motion': 0, 'static': 0}
        # Free recording state
        self.free_recording  = False
        self.free_buffer     = []
        self.free_row_count  = 0
        # Heatmap history
        self.csi_history = deque(maxlen=HISTORY)
        self._build_ui()
        self._seq_timer = QTimer()
        self._seq_timer.timeout.connect(self._tick)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        from PyQt6.QtWidgets import QProgressBar, QStackedWidget
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        root.addWidget(self._stack)

        # ════════════════════════════════════════════════════════════════════
        # PAGE 0 — REGISTRAZIONE LIBERA (vista principale)
        # ════════════════════════════════════════════════════════════════════
        page_rec = QWidget()
        p0 = QVBoxLayout(page_rec)
        p0.setContentsMargins(16, 12, 16, 12)
        p0.setSpacing(10)

        # Header
        hdr0 = QHBoxLayout()
        lbl0 = QLabel("DATI LIVE"); lbl0.setObjectName("muted")
        hdr0.addWidget(lbl0); hdr0.addStretch()
        self.btn_free_rec = QPushButton("● REGISTRA CSV")
        self.btn_free_rec.setStyleSheet(
            f"color: {ACCENT_GREEN}; border: 1px solid {ACCENT_GREEN}; border-radius: 5px; padding: 4px 10px;")
        self.btn_free_rec.clicked.connect(self._toggle_free_record)
        hdr0.addWidget(self.btn_free_rec)
        self.btn_free_save = QPushButton("SALVA CSV")
        self.btn_free_save.setObjectName("danger")
        self.btn_free_save.setEnabled(False)
        self.btn_free_save.clicked.connect(self._save_free_csv)
        hdr0.addWidget(self.btn_free_save)
        btn_go_train = QPushButton("TRAINING  →")
        btn_go_train.setStyleSheet(
            f"color: {ACCENT_BLUE}; border: 1px solid {ACCENT_BLUE}; border-radius: 5px; padding: 4px 12px;")
        btn_go_train.clicked.connect(lambda: self._stack.setCurrentIndex(1))
        hdr0.addWidget(btn_go_train)
        p0.addLayout(hdr0)

        # Splitter orizzontale: heatmap | tabella
        splitter0 = QSplitter(Qt.Orientation.Horizontal)

        hm_card = QFrame(); hm_card.setObjectName("card")
        hml = QVBoxLayout(hm_card); hml.setContentsMargins(10, 10, 10, 8)
        hml.addWidget(info_row("HEATMAP 64 SOTTOPORTANTI",
            "Mappa 2D dell'<b>ampiezza CSI nel tempo</b> per tutte le sottoportanti.<br><br>"
            "<b>Asse X</b> — frame nel tempo (ultimi 200)<br>"
            "<b>Asse Y</b> — indice sottoportante (0–63)<br>"
            "<b>Colori</b> — dal nero (basso) al giallo/bianco (alto)<br><br>"
            "<u>Colonne verticali multicolore</u> = disturbo/movimento."))
        self.canvas_hm = DarkCanvas((5, 3))
        self.ax_hm = self.canvas_hm.fig.add_subplot(111)
        self.canvas_hm.fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.12)
        hml.addWidget(self.canvas_hm, 1)
        splitter0.addWidget(hm_card)

        tbl_card = QFrame(); tbl_card.setObjectName("card")
        tbll = QVBoxLayout(tbl_card); tbll.setContentsMargins(10, 10, 10, 8)
        tbll.addWidget(info_row("DATI LIVE",
            "Tabella degli <b>ultimi 50 pacchetti CSI</b> ricevuti.<br><br>"
            "<b>RSSI</b> — potenza segnale (<b>dBm</b>)<br>"
            "<b>DELTA</b> — variazione media di ampiezza<br><br>"
            "<u>Delta basso e stabile</u> = stanza ferma."))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "MAC", "RSSI", "DELTA", "TIMESTAMP"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        tbll.addWidget(self.table)
        splitter0.addWidget(tbl_card)

        splitter0.setSizes([500, 400])
        p0.addWidget(splitter0, 1)

        self._stack.addWidget(page_rec)   # index 0

        # ════════════════════════════════════════════════════════════════════
        # PAGE 1 — TRAINING GUIDATO (schermata a pieno schermo)
        # ════════════════════════════════════════════════════════════════════
        page_train = QWidget()
        p1 = QVBoxLayout(page_train)
        p1.setContentsMargins(32, 20, 32, 20)
        p1.setSpacing(16)

        # Header training
        hdr1 = QHBoxLayout()
        btn_back = QPushButton("←  INDIETRO")
        btn_back.setStyleSheet(
            f"color: {TEXT_MUTED}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 12px;")
        btn_back.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        hdr1.addWidget(btn_back); hdr1.addStretch()
        lbl1 = QLabel("RACCOLTA DATI DI TRAINING"); lbl1.setObjectName("muted")
        hdr1.addWidget(lbl1); hdr1.addStretch()
        self.btn_save_train = QPushButton("SALVA TRAINING CSV")
        self.btn_save_train.setObjectName("danger")
        self.btn_save_train.setEnabled(False)
        self.btn_save_train.clicked.connect(self._save_train_csv)
        hdr1.addWidget(self.btn_save_train)
        self.btn_train_model = QPushButton("▶  ADDESTRA MODELLO")
        self.btn_train_model.setObjectName("success")
        self.btn_train_model.setEnabled(False)
        self.btn_train_model.clicked.connect(self._train_model)
        hdr1.addWidget(self.btn_train_model)
        p1.addLayout(hdr1)

        # Instruction card
        instr_card = QFrame(); instr_card.setObjectName("card")
        il = QVBoxLayout(instr_card); il.setContentsMargins(36, 28, 36, 28); il.setSpacing(14)

        self.lbl_phase = QLabel("PREMI AVVIA PER INIZIARE")
        self.lbl_phase.setStyleSheet(
            f"color: {ACCENT_BLUE}; font-size: 22px; font-weight: bold; letter-spacing: 3px;")
        self.lbl_phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        il.addWidget(self.lbl_phase)

        self.lbl_instr = QLabel(
            "Il training raccoglie 3 fasi da 2 minuti ciascuna:\n"
            "stanza vuota · movimento · presenza statica\n\n"
            "STATICA: cambia punto nella stanza ogni ~30s (TX, centro, RX, lati).\n"
            "Cambiare solo postura nello stesso punto non basta.\n\n"
            "Nota: il rumore da altre stanze è normale — il modello\n"
            "imparerà a distinguerlo dal segnale della tua stanza."
        )
        self.lbl_instr.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px;")
        self.lbl_instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_instr.setWordWrap(True)
        il.addWidget(self.lbl_instr)

        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setStyleSheet(
            f"color: {ACCENT_AMBER}; font-size: 64px; font-weight: bold;")
        self.lbl_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        il.addWidget(self.lbl_countdown)
        p1.addWidget(instr_card, 1)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setTextVisible(False); self.progress.setFixedHeight(12)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 6px; }}
            QProgressBar::chunk {{ background: {ACCENT_GREEN}; border-radius: 6px; }}
        """)
        p1.addWidget(self.progress)

        # Stats row
        stats = QHBoxLayout()
        self.lbl_phase_stat = self._stat("FASE",      "—")
        self.lbl_pkt_stat   = self._stat("PACCHETTI", "0")
        self.lbl_total_stat = self._stat("TOTALE",    "0")
        self.lbl_delta_stat = self._stat("DELTA CSI", "—")
        for w, _ in [self.lbl_phase_stat, self.lbl_pkt_stat,
                     self.lbl_total_stat, self.lbl_delta_stat]:
            stats.addWidget(w)
        p1.addLayout(stats)

        # Contatori per etichetta
        counts_row = QHBoxLayout(); counts_row.setSpacing(16)
        self.lbl_cnt_empty  = self._count_lbl("VUOTA",    "0")
        self.lbl_cnt_motion = self._count_lbl("MOVIMENTO","0")
        self.lbl_cnt_static = self._count_lbl("STATICA",  "0")
        counts_row.addStretch()
        for w, _ in (self.lbl_cnt_empty, self.lbl_cnt_motion, self.lbl_cnt_static):
            counts_row.addWidget(w)
        counts_row.addStretch()
        p1.addLayout(counts_row)

        # Avvia / Interrompi buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(12)
        self.btn_start = QPushButton("▶  AVVIA TRAINING")
        self.btn_start.setFixedHeight(52)
        self.btn_start.setStyleSheet(
            f"color: {ACCENT_GREEN}; border: 2px solid {ACCENT_GREEN};"
            f" font-size: 14px; letter-spacing: 2px; border-radius: 6px;")
        self.btn_start.clicked.connect(self._start)
        btn_row.addWidget(self.btn_start, 3)

        self.btn_abort = QPushButton("■  INTERROMPI")
        self.btn_abort.setFixedHeight(52)
        self.btn_abort.setStyleSheet(
            f"color: {ACCENT_RED}; border: 2px solid {ACCENT_RED};"
            f" font-size: 14px; letter-spacing: 2px; border-radius: 6px;")
        self.btn_abort.setVisible(False)
        self.btn_abort.clicked.connect(self._confirm_abort)
        btn_row.addWidget(self.btn_abort, 1)
        p1.addLayout(btn_row)

        self._stack.addWidget(page_train)  # index 1

    def _stat(self, label, value):
        f = QFrame(); f.setObjectName("card"); f.setFixedHeight(64)
        vl = QVBoxLayout(f); vl.setContentsMargins(14, 8, 14, 8)
        ll = QLabel(label); ll.setObjectName("muted")
        lv = QLabel(value)
        lv.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 16px; font-weight: bold;")
        vl.addWidget(ll); vl.addWidget(lv)
        return f, lv

    def _count_lbl(self, label, value):
        w = QWidget()
        hl = QHBoxLayout(w); hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
        ll = QLabel(f"{label}:")
        ll.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        lv = QLabel(value)
        lv.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 13px; font-weight: bold;")
        hl.addWidget(ll); hl.addWidget(lv)
        return w, lv

    # ── Training logic ───────────────────────────────────────────────────────
    def _start(self):
        self.all_rows     = []
        self.pkt_count    = 0
        self.phase_idx    = -1
        self.phase_counts = {'empty': 0, 'motion': 0, 'static': 0}
        self._update_count_labels()
        self.btn_start.setEnabled(False)
        self.btn_save_train.setEnabled(False)
        self.btn_train_model.setEnabled(False)
        self.btn_abort.setVisible(True)
        self._next_phase()

    def _confirm_abort(self):
        from PyQt6.QtWidgets import QMessageBox
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Interrompi training")
        dlg.setText("Interrompere il training?\nTutti i dati raccolti finora verranno eliminati.")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        dlg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        dlg.button(QMessageBox.StandardButton.Yes).setText("Sì, interrompi")
        dlg.button(QMessageBox.StandardButton.Cancel).setText("Annulla")
        if dlg.exec() == QMessageBox.StandardButton.Yes:
            self._abort_training()

    def _abort_training(self):
        self._seq_timer.stop()
        self.phase_idx   = -1
        self.phase_state = None
        self.time_left   = 0
        self.buffer      = []
        self.all_rows    = []
        self.lbl_phase.setText("PREMI AVVIA PER INIZIARE")
        self.lbl_phase.setStyleSheet(
            f"color: {ACCENT_BLUE}; font-size: 22px; font-weight: bold; letter-spacing: 3px;")
        self.lbl_instr.setText(
            "Il training raccoglie 3 fasi da 2 minuti ciascuna:\n"
            "stanza vuota · movimento · presenza statica\n\n"
            "STATICA: cambia punto nella stanza ogni ~30s (TX, centro, RX, lati).\n"
            "Cambiare solo postura nello stesso punto non basta.\n\n"
            "Nota: il rumore da altre stanze è normale — il modello\n"
            "imparerà a distinguerlo dal segnale della tua stanza."
        )
        self.lbl_countdown.setText("")
        self.progress.setValue(0)
        self._set_progress_color(ACCENT_GREEN)
        self.lbl_phase_stat[1].setText("—")
        self.lbl_pkt_stat[1].setText("0")
        self.lbl_total_stat[1].setText("0")
        self.lbl_delta_stat[1].setText("—")
        self.phase_counts = {'empty': 0, 'motion': 0, 'static': 0}
        self._update_count_labels()
        self.btn_start.setEnabled(True)
        self.btn_save_train.setEnabled(False)
        self.btn_train_model.setEnabled(False)
        self.btn_abort.setVisible(False)

    def _next_phase(self):
        self.phase_idx += 1
        if self.phase_idx >= len(self.PHASES):
            self._finish()
            return
        label, title, instr, _ = self.PHASES[self.phase_idx]
        self.current_label = label
        self.phase_state   = 'prep'
        self.time_left     = self.PREP_TIME
        self.buffer        = []
        self.lbl_phase.setText(title)
        self.lbl_phase.setStyleSheet(
            f"color: {ACCENT_AMBER}; font-size: 20px; font-weight: bold; letter-spacing: 3px;")
        self.lbl_instr.setText(f"⚠  PREPARATI:\n{instr}\n\nHai {self.PREP_TIME} secondi.")
        self.progress.setValue(0)
        self._set_progress_color(ACCENT_AMBER)
        self.lbl_phase_stat[1].setText(f"{self.phase_idx+1}/3")
        threading.Thread(target=lambda: play_sound('Submarine', 1), daemon=True).start()
        self._seq_timer.start(1000)

    def _set_progress_color(self, color):
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 5px; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
        """)

    def _tick(self):
        self.time_left -= 1
        self.lbl_countdown.setText(str(self.time_left) if self.time_left > 0 else "")
        if self.phase_state == 'prep':
            _, _, _, rec_secs = self.PHASES[self.phase_idx]
            self.progress.setValue(
                int((self.PREP_TIME - self.time_left) / self.PREP_TIME * 100))
            if self.time_left <= 3:
                threading.Thread(target=lambda: play_sound('Tink', 1), daemon=True).start()
            if self.time_left <= 0:
                self._start_recording()
        elif self.phase_state == 'record':
            _, _, _, rec_secs = self.PHASES[self.phase_idx]
            elapsed = rec_secs - self.time_left
            self.progress.setValue(int(elapsed / rec_secs * 100))
            self.lbl_pkt_stat[1].setText(str(len(self.buffer)))
            self.lbl_total_stat[1].setText(str(len(self.all_rows)))
            if self.time_left <= 0:
                self._end_recording()

    def _start_recording(self):
        _, title, instr, rec_secs = self.PHASES[self.phase_idx]
        self.phase_state = 'record'
        self.time_left   = rec_secs
        self.buffer      = []
        self.lbl_phase.setStyleSheet(
            f"color: {ACCENT_GREEN}; font-size: 20px; font-weight: bold; letter-spacing: 3px;")
        self.lbl_instr.setText(f"● REGISTRAZIONE IN CORSO\n\n{instr}")
        self._set_progress_color(ACCENT_GREEN)
        threading.Thread(target=lambda: play_sound('Glass', 2, 0.2), daemon=True).start()

    def _end_recording(self):
        self._seq_timer.stop()
        self.all_rows.extend(self.buffer)
        threading.Thread(target=lambda: play_sound('Hero', 1), daemon=True).start()
        self.lbl_countdown.setText("✓")
        self.lbl_instr.setText(f"Fase completata — {len(self.buffer)} pacchetti registrati.")
        QTimer.singleShot(2500, self._next_phase)

    def _finish(self):
        self._seq_timer.stop()
        self.phase_idx   = -1
        self.phase_state = None
        self.lbl_phase.setText("TRAINING COMPLETATO")
        self.lbl_phase.setStyleSheet(
            f"color: {ACCENT_GREEN}; font-size: 20px; font-weight: bold; letter-spacing: 3px;")
        self.lbl_instr.setText(
            f"Raccolti {len(self.all_rows)} pacchetti in 3 fasi.\n"
            f"Premi SALVA TRAINING CSV per esportare il dataset.")
        self.lbl_countdown.setText("")
        self.progress.setValue(100)
        self.btn_start.setEnabled(True)
        self.btn_save_train.setEnabled(len(self.all_rows) > 0)
        self.btn_train_model.setEnabled(len(self.all_rows) > 0)
        self.btn_abort.setVisible(False)
        threading.Thread(target=lambda: play_sound('Blow', 3, 0.3), daemon=True).start()

    def _update_count_labels(self):
        self.lbl_cnt_empty[1].setText(str(self.phase_counts.get('empty',  0)))
        self.lbl_cnt_motion[1].setText(str(self.phase_counts.get('motion', 0)))
        self.lbl_cnt_static[1].setText(str(self.phase_counts.get('static', 0)))

    def _train_model(self):
        if not self.all_rows:
            return
        self.btn_train_model.setEnabled(False)
        self.btn_train_model.setText("ADDESTRAMENTO...")

        def _run():
            try:
                import joblib
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import confusion_matrix, accuracy_score

                X, y = [], []
                for row in self.all_rows:
                    # row: [timestamp, label, mac, rssi, delta, amp_0..63, phase_0..63]
                    label = row[1]
                    rssi  = float(row[3])
                    delta = float(row[4])
                    amps  = [float(v) for v in row[5:69]]
                    phases= [float(v) for v in row[69:133]]
                    X.append(amps + phases + [delta, rssi])
                    y.append(label)

                X = np.array(X); y = np.array(y)
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(X)
                clf    = RandomForestClassifier(
                    n_estimators=150, max_depth=12,
                    class_weight='balanced', random_state=42, n_jobs=-1)
                # Cross-val veloce (3 fold)
                cv_scores = cross_val_score(clf, X_sc, y, cv=3, scoring='accuracy')
                clf.fit(X_sc, X_sc)  # dummy — poi fit reale
                clf.fit(X_sc, y)
                y_pred = clf.predict(X_sc)
                acc    = accuracy_score(y, y_pred)
                cm     = confusion_matrix(y, clf.predict(X_sc), labels=clf.classes_)
                features = ['amp_%d'%i for i in range(64)] + \
                           ['phase_%d'%i for i in range(64)] + ['delta','rssi']
                joblib.dump(clf,    _MODEL_DIR / 'csi_model.pkl')
                joblib.dump(scaler, _MODEL_DIR / 'csi_scaler.pkl')
                with open(_MODEL_DIR / 'csi_features.json', 'w') as f:
                    json.dump(features, f)
                # Ricarica il modello globale
                global ML_MODEL, ML_SCALER, ML_FEATURES
                ML_MODEL, ML_SCALER, ML_FEATURES = _load_model()
                return acc, cv_scores, cm, clf.classes_, None
            except Exception as e:
                return None, None, None, None, str(e)

        def _done(result):
            acc, cv_scores, cm, classes, err = result
            self.btn_train_model.setEnabled(True)
            self.btn_train_model.setText("▶  ADDESTRA MODELLO")
            if err:
                QMessageBox.critical(self, "Errore training", str(err))
                return
            # Dialog risultati
            dlg = QDialog(self)
            dlg.setWindowTitle("Risultati Training")
            dlg.setMinimumWidth(500)
            dlg.setStyleSheet(f"""
                QDialog, QWidget {{ background:{BG_DARK}; color:{TEXT_PRIMARY};
                    font-family:'SF Mono','Menlo',monospace; font-size:12px; }}
                QPushButton {{ background:transparent; color:{ACCENT_BLUE};
                    border:1px solid {ACCENT_BLUE}; padding:6px 18px; border-radius:4px; }}
            """)
            vl = QVBoxLayout(dlg)
            # Accuracy
            row_acc = QHBoxLayout()
            for label_txt, val_txt in [
                ("ACCURACY (train)", f"{acc*100:.1f}%"),
                ("CV MEDIO (3-fold)", f"{cv_scores.mean()*100:.1f}%"),
                ("CV STD",            f"±{cv_scores.std()*100:.1f}%"),
            ]:
                card = QFrame(); card.setObjectName("card")
                cl = QVBoxLayout(card); cl.setContentsMargins(14,8,14,8)
                ll = QLabel(label_txt); ll.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;")
                lv = QLabel(val_txt)
                lv.setStyleSheet(f"color:{ACCENT_GREEN};font-size:18px;font-weight:bold;")
                cl.addWidget(ll); cl.addWidget(lv)
                row_acc.addWidget(card)
            vl.addLayout(row_acc)
            # Campioni per classe
            cnt_lbl = "  ·  ".join(
                f"{c}: {int((np.array(y)==c).sum())}" for c in classes
                if 'y' in dir()
            )
            # Confusion matrix testuale
            cm_header = "MATRICE DI CONFUSIONE\n"
            cm_header += "             " + "  ".join(f"{c:>8}" for c in classes) + "\n"
            for i, c in enumerate(classes):
                cm_header += f"  {c:>10}:  " + "  ".join(f"{cm[i,j]:>8}" for j in range(len(classes))) + "\n"
            lbl_cm = QLabel(cm_header)
            lbl_cm.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:11px; font-family:monospace;")
            lbl_cm.setWordWrap(False)
            card_cm = QFrame(); card_cm.setObjectName("card")
            cm_layout = QVBoxLayout(card_cm); cm_layout.setContentsMargins(16,12,16,12)
            cm_layout.addWidget(lbl_cm)
            vl.addWidget(card_cm)
            # Campioni
            # Conta da all_rows
            counts = {}
            for row in self.all_rows:
                lbl = row[1]; counts[lbl] = counts.get(lbl, 0) + 1
            cnt_str = "  ·  ".join(f"{k}: {v}" for k,v in sorted(counts.items()))
            lbl_cnt = QLabel(f"Campioni: {cnt_str}")
            lbl_cnt.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
            vl.addWidget(lbl_cnt)
            note = QLabel("Modello salvato. Verrà usato immediatamente per la classificazione.")
            note.setStyleSheet(f"color:{ACCENT_GREEN}; font-size:11px;")
            vl.addWidget(note)
            btn_ok = QPushButton("CHIUDI")
            btn_ok.clicked.connect(dlg.accept)
            vl.addWidget(btn_ok)
            dlg.exec()

        # Esegui in thread separato per non bloccare la UI
        result_holder = [None]
        def _thread_fn():
            result_holder[0] = _run()
            # Schedula il callback sul thread principale
            QTimer.singleShot(0, lambda: _done(result_holder[0]))
        threading.Thread(target=_thread_fn, daemon=True).start()

    def _save_train_csv(self):
        if not self.all_rows: return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva Dataset Training",
            str(DATA_DIR / f"csi_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'label', 'mac', 'rssi', 'delta'] +
                            [f'amp_{i}'   for i in range(64)] +
                            [f'phase_{i}' for i in range(64)])
                for row in self.all_rows:
                    w.writerow(row)

    # ── Free recording logic ─────────────────────────────────────────────────
    def _toggle_free_record(self):
        self.free_recording = not self.free_recording
        if self.free_recording:
            self.free_buffer = []
            self.btn_free_rec.setText("■ STOP REGISTRAZIONE")
            self.btn_free_rec.setStyleSheet(
                f"color: {ACCENT_RED}; border: 1px solid {ACCENT_RED}; border-radius: 5px;")
            self.btn_free_save.setEnabled(False)
        else:
            self.btn_free_rec.setText("● REGISTRA CSV")
            self.btn_free_rec.setStyleSheet(
                f"color: {ACCENT_GREEN}; border: 1px solid {ACCENT_GREEN}; border-radius: 5px;")
            self.btn_free_save.setEnabled(len(self.free_buffer) > 0)

    def _save_free_csv(self):
        if not self.free_buffer: return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva Dataset",
            str(DATA_DIR / f"csi_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'mac', 'rssi', 'delta'] +
                            [f'amp_{i}'   for i in range(64)] +
                            [f'phase_{i}' for i in range(64)])
                for row in self.free_buffer:
                    w.writerow(row)

    # ── Data ingestion ───────────────────────────────────────────────────────
    def update_data(self, parsed, delta=0.0):
        amp   = parsed['amp']
        phase = parsed.get('phase', np.zeros(64))
        rssi  = parsed['rssi']
        mac   = parsed['mac']
        n     = min(len(amp), 64)
        amp_p   = list(amp[:n])   + [0.0] * (64 - n)
        phase_p = list(phase[:n]) + [0.0] * (64 - n)

        # Training buffer
        if self.phase_state == 'record':
            self.buffer.append(
                [datetime.now().isoformat(), self.current_label,
                 mac, rssi, round(delta, 3)] +
                [round(v, 2)  for v in amp_p] +
                [round(v, 4)  for v in phase_p]
            )
            self.lbl_delta_stat[1].setText(f"{delta:.2f}")
            if self.current_label in self.phase_counts:
                self.phase_counts[self.current_label] += 1
                self._update_count_labels()

        # Free recording buffer
        if self.free_recording:
            self.free_buffer.append(
                [datetime.now().isoformat(), mac, rssi, round(delta, 3)] +
                [round(v, 2)  for v in amp_p] +
                [round(v, 4)  for v in phase_p]
            )

        # Heatmap history (draw avviene in render(), non qui)
        self.csi_history.append(amp)

        # Live table (last 50 rows)
        if self.table.rowCount() >= 50:
            self.table.removeRow(0)
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(self.free_row_count)))
        self.table.setItem(r, 1, QTableWidgetItem(mac))
        self.table.setItem(r, 2, QTableWidgetItem(f"{rssi} dBm"))
        self.table.setItem(r, 3, QTableWidgetItem(f"{delta:.2f}"))
        self.table.setItem(r, 4, QTableWidgetItem(datetime.now().strftime("%H:%M:%S.%f")[:-3]))
        self.table.scrollToBottom()
        self.free_row_count += 1

    def render(self):
        """Chiamato dal render_slow_timer (~10 fps). Ridisegna solo la heatmap."""
        if len(self.csi_history) < 10:
            return
        min_len = min(len(x) for x in self.csi_history)
        data    = np.array([x[:min_len] for x in self.csi_history])
        n       = min(data.shape[1], 64)
        data    = data[:, :n]
        self.ax_hm.cla()
        self.ax_hm.set_facecolor(BG_CARD)
        self.ax_hm.tick_params(colors=TEXT_MUTED, labelsize=8)
        for sp in self.ax_hm.spines.values(): sp.set_edgecolor(BORDER)
        self.ax_hm.imshow(data.T, aspect='auto', cmap='inferno',
                          interpolation='nearest', origin='lower')
        self.ax_hm.set_xlabel('Frame', color=TEXT_MUTED, fontsize=8)
        self.ax_hm.set_ylabel('Sottoportante', color=TEXT_MUTED, fontsize=8)
        self.canvas_hm.draw_idle()


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — FASE & TOF
# ══════════════════════════════════════════════════════════════════════════
# Spaziatura sottoportanti 802.11n HT20 = 312.5 kHz
SUBCARRIER_SPACING = 312.5e3
C_LIGHT            = 3e8

class PhaseTab(QWidget):
    def __init__(self):
        super().__init__()
        self.prev_phase    = None
        self.tof_history   = deque(maxlen=HISTORY)
        self.slope_history = deque(maxlen=HISTORY)
        self.var_accum     = None
        self.n_frames      = 0
        self.tof_smooth    = None   # EMA del τ — usato dalla mappa per le ellissi
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Metriche (PERCORSO EXTRA rimosso: troppo rumoroso su ESP32 senza cal. hardware)
        metrics = QHBoxLayout()
        self.lbl_tof   = self._metric("RITARDO STIMATO τ", "— ns",  metrics)
        self.lbl_slope = self._metric("PENDENZA FASE",     "— rad", metrics)
        self.lbl_pvar  = self._metric("VAR. FASE",         "—",     metrics)
        layout.addLayout(metrics)

        # Grafici riga 1: spettro fase + differenza inter-frame
        row1 = QHBoxLayout(); row1.setSpacing(12)

        c1 = self._card()
        l1 = QVBoxLayout(c1); l1.setContentsMargins(12, 12, 12, 12)
        l1.addWidget(info_row("FASE UNWRAPPED — FRAME CORRENTE",
            "Profilo di fase del pacchetto CSI più recente, dopo l'<b>unwrapping</b>"
            " (rimozione dei salti artificiali di ±π).<br><br>"
            "<b>Asse X</b> — indice sottoportante (0–63)<br>"
            "<b>Asse Y</b> — fase in <b>radianti</b> (range tipico ±10 rad dopo unwrap)<br>"
            "<b>Linea blu</b> — fase misurata<br>"
            "<b>Tratteggio arancione</b> — retta di fit lineare<br><br>"
            "<u>Come si legge:</u> in un canale pulito la fase è quasi lineare."
            " <u>La pendenza della retta</u> codifica il <b>ritardo del percorso"
            " dominante (τ)</b>. Deviazioni dalla retta indicano"
            " componenti di <b>multipath</b> o rumore."))
        self.canvas_ph = DarkCanvas((5, 2.4))
        self.ax_ph = self.canvas_ph.fig.add_subplot(111)
        self.canvas_ph.fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14)
        style_ax(self.ax_ph)
        l1.addWidget(self.canvas_ph)
        row1.addWidget(c1)

        c2 = self._card()
        l2 = QVBoxLayout(c2); l2.setContentsMargins(12, 12, 12, 12)
        l2.addWidget(info_row("DIFFERENZA DI FASE (Δφ INTER-FRAME)",
            "Differenza di fase tra frame corrente e precedente, con"
            " <b>rimozione del drift CFO</b> (sottrazione della mediana).<br>"
            "Il CFO dell'ESP32 introduce uno shift verticale casuale su tutti"
            " i subcarrier ad ogni pacchetto — la mediana lo cattura e lo rimuove.<br><br>"
            "<b>Asse X</b> — indice sottoportante (0–63)<br>"
            "<b>Asse Y</b> — Δrad corretto (stanza ferma → piatto a zero)<br><br>"
            "<u>Come si legge:</u> piatto vicino allo zero = nessun movimento."
            " Pattern ondulato = qualcuno si sta muovendo."))
        self.canvas_dp = DarkCanvas((5, 2.4))
        self.ax_dp = self.canvas_dp.fig.add_subplot(111)
        self.canvas_dp.fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14)
        style_ax(self.ax_dp)
        l2.addWidget(self.canvas_dp)
        row1.addWidget(c2)
        layout.addLayout(row1)

        # Grafici riga 2: ToF nel tempo + varianza accumulata per subcarrier
        row2 = QHBoxLayout(); row2.setSpacing(12)

        c3 = self._card()
        l3 = QVBoxLayout(c3); l3.setContentsMargins(12, 12, 12, 12)
        l3.addWidget(info_row("STIMA τ NEL TEMPO (proxy multipath)",
            "Ritardo di gruppo del percorso dominante stimato nel tempo,"
            " calcolato dalla pendenza della fase unwrapped:<br>"
            "<b>τ = −slope / (2π · Δf)</b>,  Δf = 312.5 kHz<br><br>"
            "<b>Asse X</b> — frame ricevuti (ultimi 200)<br>"
            "<b>Asse Y</b> — ritardo stimato in <b>nanosecondi (ns)</b><br><br>"
            "<u>Come si legge:</u> il valore di base corrisponde al"
            " <u>percorso diretto TX→RX</u>. Quando una persona si muove"
            " il multipath cambia e <u>τ oscilla attorno alla baseline</u>."
            " Nello Step 2 questo valore diventa il <b>semiasse maggiore"
            " delle ellissi</b> sulla mappa."))
        self.canvas_tof = DarkCanvas((5, 2.4))
        self.ax_tof = self.canvas_tof.fig.add_subplot(111)
        self.canvas_tof.fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14)
        style_ax(self.ax_tof)
        l3.addWidget(self.canvas_tof)
        row2.addWidget(c3)

        c4 = self._card()
        l4 = QVBoxLayout(c4); l4.setContentsMargins(12, 12, 12, 12)
        l4.addWidget(info_row("VARIANZA FASE PER SUBCARRIER (attività)",
            "Varianza accumulata di <b>Δφ</b> per ciascuna sottoportante,"
            " aggiornata con <b>media mobile esponenziale</b> (α = 0.05).<br><br>"
            "<b>Asse X</b> — indice sottoportante (0–63)<br>"
            "<b>Asse Y</b> — var(Δφ) in <b>rad²</b> (cresce con l'attività)<br><br>"
            "<u>Come si legge:</u> <u>i picchi indicano le sottoportanti più"
            " sensibili al movimento</u>. Quelle con varianza alta sono"
            " 'illuminate' dalla riflessione del corpo in movimento."
            " Nello Step 2 questo profilo serve a <b>pesare i contributi"
            " nella stima della posizione</b> sull'ellisse."))
        self.canvas_var = DarkCanvas((5, 2.4))
        self.ax_var = self.canvas_var.fig.add_subplot(111)
        self.canvas_var.fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.14)
        style_ax(self.ax_var)
        l4.addWidget(self.canvas_var)
        row2.addWidget(c4)
        layout.addLayout(row2)

    def _card(self):
        f = QFrame(); f.setObjectName("card"); return f

    def _lbl_muted(self, text):
        l = QLabel(text); l.setObjectName("muted"); return l

    def _metric(self, label, value, parent):
        f = self._card(); f.setFixedHeight(80)
        vl = QVBoxLayout(f); vl.setContentsMargins(16, 12, 16, 12)
        ll = QLabel(label); ll.setObjectName("muted")
        lv = QLabel(value); lv.setObjectName("value")
        lv.setFont(QFont("SF Mono", 18, QFont.Weight.Bold))
        vl.addWidget(ll); vl.addWidget(lv)
        parent.addWidget(f)
        return lv

    # ── Logica ──────────────────────────────────────────────────────────────
    def update_data(self, parsed):
        """Aggiorna solo lo stato numpy — nessun rendering."""
        raw_phase = parsed['phase']
        if len(raw_phase) < 4:
            return

        ph = np.unwrap(raw_phase)
        x  = np.arange(len(ph))

        try:
            slope, _ = np.polyfit(x, ph, 1)
        except Exception:
            return

        tof_s    = -slope / (2.0 * np.pi * SUBCARRIER_SPACING)
        d_direct = np.sqrt((TX_POS[0]-RX_POS[0])**2 + (TX_POS[1]-RX_POS[1])**2)
        d_extra  = max(abs(tof_s) * C_LIGHT - d_direct, 0.0)

        self.tof_history.append(tof_s * 1e9)
        # EMA per smoothing (α=0.15 → risponde in ~6 frame)
        if self.tof_smooth is None:
            self.tof_smooth = tof_s * 1e9
        else:
            self.tof_smooth = 0.85 * self.tof_smooth + 0.15 * (tof_s * 1e9)
        self.slope_history.append(slope)
        self.n_frames += 1

        if self.prev_phase is not None and len(self.prev_phase) == len(ph):
            delta_ph = ph - self.prev_phase
            if self.var_accum is None or len(self.var_accum) != len(delta_ph):
                self.var_accum = delta_ph ** 2
            else:
                self.var_accum = 0.95 * self.var_accum + 0.05 * delta_ph ** 2
        else:
            delta_ph = np.zeros_like(ph)
            if self.var_accum is None:
                self.var_accum = np.zeros_like(ph)

        self.prev_phase = ph

        # Aggiorna solo le label testuali (operazione leggera)
        self.lbl_tof.setText(f"{tof_s*1e9:.1f} ns")
        self.lbl_slope.setText(f"{slope:.4f}")
        if self.var_accum is not None:
            self.lbl_pvar.setText(f"{float(np.mean(self.var_accum)):.3f}")

        # Snapshot per render()
        self._render_state = (ph, delta_ph, x, slope)

    def render(self):
        """Chiamato dal timer centrale — ridisegna solo se c'è uno snapshot nuovo."""
        if not hasattr(self, '_render_state') or self._render_state is None:
            return
        ph, delta_ph, x, slope = self._render_state
        self._render_state = None
        try:
            self._draw(ph, delta_ph, x, slope)
        except Exception:
            pass  # ignora errori di layout durante resize

    def _draw(self, ph, delta_ph, x, slope):
        # — Fase unwrapped detrended (rimuove la retta di fit per mostrare
        #   solo le deviazioni dal ritardo medio — più stabile a stanza ferma) —
        ph_detrended = ph - (slope * x + (ph[0] - slope * x[0]))
        self.ax_ph.cla(); style_ax(self.ax_ph)
        self.ax_ph.plot(x, ph_detrended, color=ACCENT_BLUE, linewidth=1.2)
        self.ax_ph.axhline(0, color=ACCENT_AMBER, linewidth=0.8,
                           linestyle='--', alpha=0.6)
        self.ax_ph.set_xlabel('Subcarrier', color=TEXT_MUTED, fontsize=8)
        self.ax_ph.set_ylabel('Δrad (detrended)', color=TEXT_MUTED, fontsize=8)
        self.canvas_ph.draw_idle()

        # — Δφ inter-frame con rimozione del drift CFO comune —
        # Sottrae la mediana per annullare lo shift verticale uniforme
        # causato dal CFO dell'ESP32 che cambia tra pacchetti
        delta_ph_clean = delta_ph - np.median(delta_ph)
        self.ax_dp.cla(); style_ax(self.ax_dp)
        self.ax_dp.fill_between(x, delta_ph_clean, alpha=0.35, color=ACCENT_GREEN)
        self.ax_dp.plot(x, delta_ph_clean, color=ACCENT_GREEN, linewidth=1)
        self.ax_dp.axhline(0, color=BORDER, linewidth=0.6)
        self.ax_dp.set_xlabel('Subcarrier', color=TEXT_MUTED, fontsize=8)
        self.ax_dp.set_ylabel('Δrad (CFO rimosso)', color=TEXT_MUTED, fontsize=8)
        self.canvas_dp.draw_idle()

        # — τ nel tempo —
        self.ax_tof.cla(); style_ax(self.ax_tof)
        if len(self.tof_history) > 1:
            y = list(self.tof_history)
            self.ax_tof.plot(y, color=ACCENT_AMBER, linewidth=1.2)
            self.ax_tof.fill_between(range(len(y)), y, alpha=0.2, color=ACCENT_AMBER)
        self.ax_tof.set_xlabel('Frame', color=TEXT_MUTED, fontsize=8)
        self.ax_tof.set_ylabel('ns', color=TEXT_MUTED, fontsize=8)
        self.canvas_tof.draw_idle()

        # — Varianza per subcarrier —
        self.ax_var.cla(); style_ax(self.ax_var)
        if self.var_accum is not None:
            self.ax_var.fill_between(x, self.var_accum, alpha=0.45, color=ACCENT_RED)
            self.ax_var.plot(x, self.var_accum, color=ACCENT_RED, linewidth=1)
        self.ax_var.set_xlabel('Subcarrier', color=TEXT_MUTED, fontsize=8)
        self.ax_var.set_ylabel('var(Δφ)', color=TEXT_MUTED, fontsize=8)
        self.canvas_var.draw_idle()


# ══════════════════════════════════════════════════════════════════════════
# DIALOG IMPOSTAZIONI
# ══════════════════════════════════════════════════════════════════════════
class SettingsDialog(QDialog):
    settings_applied = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Impostazioni")
        self.setMinimumWidth(460)
        self.setStyleSheet(f"""
            QDialog, QWidget {{ background: {BG_DARK}; color: {TEXT_PRIMARY};
                font-family: 'SF Mono','Menlo',monospace; font-size: 12px; }}
            QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px;
                margin-top: 10px; padding: 10px; color: {TEXT_MUTED}; font-size: 11px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QLineEdit, QDoubleSpinBox, QSpinBox {{
                background: {BG_CARD}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 8px; }}
            QCheckBox {{ color: {TEXT_PRIMARY}; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; }}
            QPushButton {{ background: transparent; color: {ACCENT_BLUE};
                border: 1px solid {ACCENT_BLUE}; padding: 6px 18px; border-radius: 4px; }}
            QPushButton:hover {{ background: rgba(0,207,255,0.1); }}
            QLabel {{ color: {TEXT_PRIMARY}; }}
            QLabel#muted {{ color: {TEXT_MUTED}; font-size: 10px; }}
            QSlider::groove:horizontal {{ background: {BG_CARD}; height: 4px; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {ACCENT_BLUE}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT_BLUE}; border-radius: 2px; }}
        """)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Connessione ──────────────────────────────────────────────────
        grp_conn = QGroupBox("CONNESSIONE")
        fl = QFormLayout(grp_conn); fl.setSpacing(8)
        self.ed_port = QLineEdit(_CFG['port'])
        self.ed_mac  = QLineEdit(_CFG['tx_mac'])
        fl.addRow("Porta seriale RX:", self.ed_port)
        fl.addRow("MAC del TX:", self.ed_mac)
        layout.addWidget(grp_conn)

        # ── Geometria stanza ─────────────────────────────────────────────
        grp_room = QGroupBox("GEOMETRIA STANZA  (richiede riavvio)")
        fl2 = QFormLayout(grp_room); fl2.setSpacing(8)
        self.sp_rw = self._dspin(_CFG['room_w'], 1.0, 30.0)
        self.sp_rh = self._dspin(_CFG['room_h'], 1.0, 30.0)
        self.sp_tx_x = self._dspin(_CFG['tx_pos'][0], 0.0, 30.0)
        self.sp_tx_y = self._dspin(_CFG['tx_pos'][1], 0.0, 30.0)
        self.sp_rx_x = self._dspin(_CFG['rx_pos'][0], 0.0, 30.0)
        self.sp_rx_y = self._dspin(_CFG['rx_pos'][1], 0.0, 30.0)
        fl2.addRow("Larghezza stanza (m):", self.sp_rw)
        fl2.addRow("Profondità stanza (m):", self.sp_rh)
        fl2.addRow("TX  X (m):", self.sp_tx_x)
        fl2.addRow("TX  Y (m):", self.sp_tx_y)
        fl2.addRow("RX  X (m):", self.sp_rx_x)
        fl2.addRow("RX  Y (m):", self.sp_rx_y)
        layout.addWidget(grp_room)

        # ── ML / Soglia confidenza ───────────────────────────────────────
        grp_ml = QGroupBox("CLASSIFICAZIONE ML")
        ml_layout = QVBoxLayout(grp_ml); ml_layout.setSpacing(8)
        thresh_row = QHBoxLayout()
        lbl_t = QLabel("Soglia confidenza:")
        self.sld_thresh = QSlider(Qt.Orientation.Horizontal)
        self.sld_thresh.setRange(30, 95)
        self.sld_thresh.setValue(int(_CFG.get('conf_threshold', 0.60) * 100))
        self.lbl_thresh_val = QLabel(f"{self.sld_thresh.value()}%")
        self.lbl_thresh_val.setFixedWidth(40)
        self.sld_thresh.valueChanged.connect(
            lambda v: self.lbl_thresh_val.setText(f"{v}%"))
        thresh_row.addWidget(lbl_t)
        thresh_row.addWidget(self.sld_thresh)
        thresh_row.addWidget(self.lbl_thresh_val)
        ml_layout.addLayout(thresh_row)
        note = QLabel("Sotto questa soglia il rilevamento mostra INCERTO")
        note.setObjectName("muted")
        ml_layout.addWidget(note)
        layout.addWidget(grp_ml)

        # ── Notifiche ────────────────────────────────────────────────────
        grp_notif = QGroupBox("NOTIFICHE macOS")
        nl = QFormLayout(grp_notif); nl.setSpacing(8)
        self.chk_notif = QCheckBox("Attiva notifiche al cambio di stato")
        self.chk_notif.setChecked(bool(_CFG.get('notifications', True)))
        self.sp_cooldown = QSpinBox()
        self.sp_cooldown.setRange(3, 300)
        self.sp_cooldown.setValue(int(_CFG.get('notif_cooldown', 10)))
        self.sp_cooldown.setSuffix(" s")
        nl.addRow("", self.chk_notif)
        nl.addRow("Cooldown tra notifiche:", self.sp_cooldown)
        layout.addWidget(grp_notif)

        # ── Bottoni ──────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Applica")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("Annulla")
        btns.accepted.connect(self._apply)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _dspin(self, val, lo, hi):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setSingleStep(0.1); s.setDecimals(1)
        s.setValue(val); return s

    def _apply(self):
        global _CFG, PORT, TX_MAC, ML_CONF_THRESHOLD
        _CFG['port']           = self.ed_port.text().strip()
        _CFG['tx_mac']         = self.ed_mac.text().strip()
        _CFG['room_w']         = self.sp_rw.value()
        _CFG['room_h']         = self.sp_rh.value()
        _CFG['tx_pos']         = [self.sp_tx_x.value(), self.sp_tx_y.value()]
        _CFG['rx_pos']         = [self.sp_rx_x.value(), self.sp_rx_y.value()]
        _CFG['conf_threshold'] = self.sld_thresh.value() / 100.0
        _CFG['notifications']  = self.chk_notif.isChecked()
        _CFG['notif_cooldown'] = self.sp_cooldown.value()
        # Applica immediata per i parametri non geometrici
        PORT              = _CFG['port']
        TX_MAC            = _CFG['tx_mac']
        ML_CONF_THRESHOLD = _CFG['conf_threshold']
        _save_settings_to_file(_CFG)
        self.settings_applied.emit(_CFG)
        self.accept()


# ══════════════════════════════════════════════════════════════════════════
# DIALOG PROFILI
# ══════════════════════════════════════════════════════════════════════════
class ProfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profili stanza")
        self.setMinimumWidth(380)
        self.setMinimumHeight(320)
        self.setStyleSheet(f"""
            QDialog, QWidget {{ background: {BG_DARK}; color: {TEXT_PRIMARY};
                font-family: 'SF Mono','Menlo',monospace; font-size: 12px; }}
            QListWidget {{ background: {BG_CARD}; border: 1px solid {BORDER};
                border-radius: 4px; color: {TEXT_PRIMARY}; font-size: 12px; }}
            QListWidget::item:selected {{ background: rgba(0,207,255,0.15); }}
            QPushButton {{ background: transparent; color: {ACCENT_BLUE};
                border: 1px solid {ACCENT_BLUE}; padding: 6px 16px; border-radius: 4px; }}
            QPushButton:hover {{ background: rgba(0,207,255,0.1); }}
            QPushButton#danger {{ color: {ACCENT_RED}; border-color: {ACCENT_RED}; }}
            QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {TEXT_MUTED}; }}
        """)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("PROFILI SALVATI")
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(lbl)
        self.lst = QListWidget()
        layout.addWidget(self.lst, 1)
        btns = QHBoxLayout()
        self.btn_load   = QPushButton("CARICA")
        self.btn_save   = QPushButton("SALVA CORRENTE")
        self.btn_delete = QPushButton("ELIMINA")
        self.btn_delete.setObjectName("danger")
        self.btn_load.clicked.connect(self._load)
        self.btn_save.clicked.connect(self._save)
        self.btn_delete.clicked.connect(self._delete)
        for b in (self.btn_load, self.btn_save, self.btn_delete):
            btns.addWidget(b)
        layout.addLayout(btns)
        close = QPushButton("CHIUDI")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

    def _refresh(self):
        self.lst.clear()
        _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        for p in sorted(_PROFILES_DIR.iterdir()):
            if p.is_dir():
                self.lst.addItem(p.name)

    def _current(self):
        item = self.lst.currentItem()
        return item.text() if item else None

    def _save(self):
        name, ok = QInputDialog.getText(self, "Salva profilo", "Nome profilo:")
        if not ok or not name.strip():
            return
        name = name.strip().replace('/', '_')
        dest = _PROFILES_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        # Copia modello se esiste
        import shutil
        for fn in ('csi_model.pkl', 'csi_scaler.pkl', 'csi_features.json'):
            src = _MODEL_DIR / fn
            if src.exists():
                shutil.copy2(src, dest / fn)
        # Salva settings
        with open(dest / 'settings.json', 'w') as f:
            json.dump(_CFG, f, indent=2)
        self._refresh()

    def _load(self):
        name = self._current()
        if not name:
            return
        src = _PROFILES_DIR / name
        import shutil
        for fn in ('csi_model.pkl', 'csi_scaler.pkl', 'csi_features.json'):
            s = src / fn
            if s.exists():
                shutil.copy2(s, _MODEL_DIR / fn)
        cfg_path = src / 'settings.json'
        if cfg_path.exists():
            global _CFG
            with open(cfg_path) as f:
                _CFG.update(json.load(f))
            _save_settings_to_file(_CFG)
        QMessageBox.information(self, "Profilo caricato",
            f"Profilo '{name}' caricato.\nRiavvia l'app per applicare la geometria stanza.")
        self.accept()

    def _delete(self):
        name = self._current()
        if not name:
            return
        r = QMessageBox.question(self, "Elimina", f"Eliminare il profilo '{name}'?")
        if r == QMessageBox.StandardButton.Yes:
            import shutil
            shutil.rmtree(_PROFILES_DIR / name, ignore_errors=True)
            self._refresh()


# ══════════════════════════════════════════════════════════════════════════
# FINESTRA PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blind Eye — CSI Motion Sensing")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._start_serial()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QVBoxLayout(central)
        ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        hdr = QFrame(); hdr.setFixedHeight(52)
        hdr.setStyleSheet(f"background: {BG_PANEL}; border-bottom: 1px solid {BORDER};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(20,0,20,0); hl.setSpacing(16)
        title = QLabel("◈  BLIND EYE")
        title.setStyleSheet(f"color: {ACCENT_BLUE}; font-size: 13px; letter-spacing: 3px; font-weight: bold;")
        hl.addWidget(title)
        hl.addStretch()

        # Stats live: fps + RSSI
        self.lbl_hdr_fps  = QLabel("— pkt/s")
        self.lbl_hdr_rssi = QLabel("RSSI —")
        for l in (self.lbl_hdr_fps, self.lbl_hdr_rssi):
            l.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; letter-spacing: 1px;")
        hl.addWidget(self.lbl_hdr_fps)
        hl.addWidget(self.lbl_hdr_rssi)

        # Separatore
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {BORDER};"); sep.setFixedHeight(24)
        hl.addWidget(sep)

        # Pulsanti profili e impostazioni
        btn_profiles = QPushButton("⊞  PROFILI")
        btn_profiles.setStyleSheet(
            f"color: {TEXT_MUTED}; border: 1px solid {BORDER}; border-radius: 4px;"
            f" padding: 4px 12px; font-size: 10px;")
        btn_profiles.clicked.connect(self._open_profiles)
        hl.addWidget(btn_profiles)

        self.btn_notif = QPushButton("🔔")
        self.btn_notif.setFixedSize(34, 34)
        self.btn_notif.setToolTip("Attiva/disattiva notifiche macOS")
        self.btn_notif.setStyleSheet(
            f"color: {ACCENT_GREEN if _CFG.get('notifications', True) else TEXT_MUTED};"
            f" border: 1px solid {ACCENT_GREEN if _CFG.get('notifications', True) else BORDER};"
            f" border-radius: 4px; font-size: 14px;")
        self.btn_notif.clicked.connect(self._toggle_notifications)
        hl.addWidget(self.btn_notif)

        btn_settings = QPushButton("⚙")
        btn_settings.setFixedSize(34, 34)
        btn_settings.setStyleSheet(
            f"color: {TEXT_MUTED}; border: 1px solid {BORDER}; border-radius: 4px;"
            f" font-size: 14px;")
        btn_settings.clicked.connect(self._open_settings)
        hl.addWidget(btn_settings)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"color: {BORDER};"); sep2.setFixedHeight(24)
        hl.addWidget(sep2)

        self.conn_dot   = QLabel("●")
        self.conn_dot.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")
        self.conn_label = QLabel("DISCONNESSO")
        self.conn_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; letter-spacing: 1px;")
        hl.addWidget(self.conn_dot); hl.addWidget(self.conn_label)
        ml.addWidget(hdr)

        self.tabs = QTabWidget()
        self.tab_dash  = DashboardTab()
        self.tab_map   = TopViewTab()
        self.tab_data  = DataTab()
        self.tab_phase = PhaseTab()
        self.tabs.addTab(self.tab_dash,  "MONITOR LIVE")
        self.tabs.addTab(self.tab_map,   "MAPPA RADIO")
        self.tabs.addTab(self.tab_data,  "DATI & TRAINING")
        self.tabs.addTab(self.tab_phase, "FASE & TOF")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        ml.addWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        import glob as _glob
        _all_ports = sorted(_glob.glob('/dev/cu.usbserial-*'), key=len)
        _tx_port = _all_ports[0] if len(_all_ports) >= 2 else "—"
        _rx_port = _find_port()
        self.status.showMessage(
            f"RX: {_rx_port}  |  TX: {_tx_port}  |  Baud: {BAUD}  |  MAC: {TX_MAC}")

        self._queue      = []
        self._queue_lock = threading.Lock()
        self._last_ml_label = None

        # Timer dati: svuota la coda e aggiorna lo stato di tutti i tab (leggero)
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self._flush)
        self.data_timer.start(33)   # ~30 fps — solo numpy, nessun disegno

        # Timer aggiornamento stats header: 1 fps
        self._hdr_stats_timer = QTimer()
        self._hdr_stats_timer.timeout.connect(self._update_hdr_stats)
        self._hdr_stats_timer.start(1000)

        # Timer render live (DashboardTab): 20 fps
        self.render_live_timer = QTimer()
        self.render_live_timer.timeout.connect(self._render_live)
        self.render_live_timer.start(50)

        # Timer render lento (PhaseTab + MapTab): 10 fps
        self.render_slow_timer = QTimer()
        self.render_slow_timer.timeout.connect(self._render_slow)
        self.render_slow_timer.start(100)


    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.settings_applied.connect(self._on_settings_applied)
        dlg.exec()

    def _on_settings_applied(self, cfg):
        # Riavvia serial worker se la porta è cambiata
        self.worker.stop()
        time.sleep(0.2)
        self._start_serial()
        import glob as _glob
        _all_ports = sorted(_glob.glob('/dev/cu.usbserial-*'), key=len)
        _tx_port = _all_ports[0] if len(_all_ports) >= 2 else "—"
        self.status.showMessage(
            f"RX: {cfg['port']}  |  TX: {_tx_port}  |  Baud: {cfg['baud']}  |  MAC: {cfg['tx_mac']}")

    def _open_profiles(self):
        ProfileDialog(self).exec()

    def _toggle_notifications(self):
        current = _CFG.get('notifications', True)
        _CFG['notifications'] = not current
        _save_settings_to_file(_CFG)
        enabled = _CFG['notifications']
        self.btn_notif.setStyleSheet(
            f"color: {ACCENT_GREEN if enabled else TEXT_MUTED};"
            f" border: 1px solid {ACCENT_GREEN if enabled else BORDER};"
            f" border-radius: 4px; font-size: 14px;")

    def _update_hdr_stats(self):
        mh = self.tab_dash.rssi_history
        if mh:
            avg_rssi = float(np.mean(list(mh)))
            self.lbl_hdr_rssi.setText(f"RSSI {avg_rssi:.0f} dBm")
        fps = self.tab_dash.fps_count
        self.lbl_hdr_fps.setText(f"{fps} pkt/s")
        # reset fps counter ogni secondo
        self.tab_dash.fps_count = 0
        self.tab_dash.last_time = time.time()

    def _start_serial(self):
        self.worker = SerialWorker()
        self.worker.data_received.connect(self._on_data)
        self.worker.status_changed.connect(self._on_status)
        self.worker.start()

    def _on_data(self, parsed):
        with self._queue_lock:
            self._queue.append(parsed)

    def _on_status(self, msg, level):
        c = ACCENT_GREEN if level == "ok" else ACCENT_RED
        self.conn_dot.setStyleSheet(f"color: {c}; font-size: 14px;")
        self.conn_label.setStyleSheet(f"color: {c}; font-size: 10px; letter-spacing: 1px;")
        self.conn_label.setText("CONNESSO" if level == "ok" else "ERRORE")
        self.status.showMessage(msg)

    def _flush(self):
        """Svuota la coda e aggiorna solo lo stato (numpy) — nessun matplotlib."""
        with self._queue_lock:
            if not self._queue: return
            items = self._queue.copy()
            self._queue.clear()

        last_valid = None
        for p in items:
            if FILTER_MAC and p['mac'] != TX_MAC:
                continue
            self.tab_dash.update_data(p)
            self.tab_phase.update_data(p)
            last_valid = p
            mh = self.tab_dash.motion_history
            delta = float(list(mh)[-1]) if mh else 0.0
            self.tab_data.update_data(p, delta)
            # Classificazione ML real-time
            ml_label, ml_conf = classify_packet(p, delta)
            if ml_label:
                # Soglia confidenza
                if ml_conf < ML_CONF_THRESHOLD:
                    display_name  = "INCERTO"
                    display_color = TEXT_MUTED
                else:
                    display_name, display_color = ML_LABELS.get(ml_label, (ml_label, TEXT_MUTED))
                self.tab_dash.update_ml(display_name, display_color, ml_conf, delta)
                # Notifica macOS al cambio di stato
                if ml_label != self._last_ml_label and ml_conf >= ML_CONF_THRESHOLD:
                    threading.Thread(
                        target=notify_mac,
                        args=(display_name, f"conf {ml_conf*100:.0f}%  ·  delta {delta:.2f}"),
                        daemon=True
                    ).start()
                self._last_ml_label = ml_label

        if last_valid is not None:
            mh       = self.tab_dash.motion_history
            activity = list(mh)[-1] / 20.0 if mh else 0.0
            # Salva snapshot mappa senza ridisegnare
            self._map_snapshot = (last_valid, activity)

    def _render_live(self):
        """20 fps — ridisegna solo se il tab Monitor Live è visibile."""
        if self.tabs.currentWidget() is self.tab_dash:
            self.tab_dash.render()

    def _render_slow(self):
        """10 fps — ridisegna solo il tab attualmente visibile."""
        current = self.tabs.currentWidget()
        if current is self.tab_phase:
            self.tab_phase.render()
        elif current is self.tab_data:
            self.tab_data.render()
        elif current is self.tab_map:
            if hasattr(self, '_map_snapshot') and self._map_snapshot is not None:
                p, activity = self._map_snapshot
                self._map_snapshot = None
                self.tab_map.update_data(p, activity,
                                         tof_ns=self.tab_phase.tof_smooth)

    def _on_tab_changed(self, _index):
        """Forza un render immediato quando si cambia tab, così non appare stale."""
        current = self.tabs.currentWidget()
        if current is self.tab_dash:
            self.tab_dash.render()
        elif current is self.tab_phase:
            self.tab_phase.render()
        elif current is self.tab_data:
            self.tab_data.render()
        elif current is self.tab_map:
            if hasattr(self, '_map_snapshot') and self._map_snapshot is not None:
                p, activity = self._map_snapshot
                self._map_snapshot = None
                self.tab_map.update_data(p, activity,
                                         tof_ns=self.tab_phase.tof_smooth)

    def resizeEvent(self, event):
        """Debounce resize: aspetta 150ms prima di ridisegnare i canvas."""
        super().resizeEvent(event)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._on_resize_done)
        self._resize_timer.start(150)

    def _on_resize_done(self):
        current = self.tabs.currentWidget()
        if current is self.tab_dash:
            self.tab_dash.render()
        elif current is self.tab_phase:
            self.tab_phase.render()
        elif current is self.tab_data:
            self.tab_data.render()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,        QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText,    QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base,          QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_PANEL))
    palette.setColor(QPalette.ColorRole.Text,          QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button,        QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.ButtonText,    QColor(TEXT_PRIMARY))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
