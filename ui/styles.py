"""
ui/styles.py
============
Dark QSS stylesheet shared by the whole app.
"""

QSS = """
QMainWindow, QWidget {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #333355;
    border-radius: 4px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: bold;
    color: #9999cc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QListWidget {
    background: #16213e;
    border: 1px solid #333355;
    border-radius: 3px;
    padding: 2px;
}
QListWidget::item { padding: 3px 6px; border-radius: 2px; }
QListWidget::item:selected { background: #0f3460; color: white; }
QTableWidget {
    background: #16213e;
    gridline-color: #252545;
    border: 1px solid #333355;
    border-radius: 3px;
    alternate-background-color: #1c1c38;
}
QTableWidget::item { padding: 4px 8px; }
QTableWidget::item:selected { background: #0f3460; }
QHeaderView::section {
    background: #0f3460;
    color: white;
    padding: 5px 10px;
    border: none;
    font-weight: bold;
}
QPushButton {
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
}
QPushButton:hover    { background: #1a4470; }
QPushButton:pressed  { background: #082030; }
QPushButton:disabled { background: #252538; color: #555577; }
QPushButton#transportPlay { padding: 0; }
QPushButton#analyze, QPushButton#load {
    background: #e94560;
    font-size: 12pt;
    padding: 8px 24px;
}
QPushButton#analyze:hover, QPushButton#load:hover { background: #c03050; }
QPushButton#analyze:disabled, QPushButton#load:disabled { background: #4a2030; color: #886688; }
QLabel        { color: #e0e0e0; }
QLabel#key    { font-size: 22pt; font-weight: bold; color: #ffdd88; }
QLabel#subkey { font-size: 9pt;  color: #9999aa; }
QDoubleSpinBox, QSpinBox, QComboBox {
    background: #16213e;
    color: #e0e0e0;
    border: 1px solid #333355;
    border-radius: 3px;
    padding: 3px 6px;
    min-width: 72px;
}
QProgressBar {
    background: #16213e;
    border: 1px solid #333355;
    border-radius: 3px;
    height: 14px;
    text-align: center;
    color: white;
    font-size: 8pt;
}
QProgressBar::chunk { background: #e94560; border-radius: 2px; }
QScrollBar:vertical { background: #16213e; width: 10px; border: none; }
QScrollBar::handle:vertical {
    background: #333355;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QRadioButton { color: #e0e0e0; spacing: 4px; }
QRadioButton::indicator { width: 12px; height: 12px; }
QSlider::groove:horizontal {
    background: #16213e;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e94560;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QStatusBar { color: #aaaacc; background: #16213e; }
QTabWidget::pane { border: 1px solid #333355; background: #1a1a2e; }
QTabBar::tab {
    background: #16213e;
    color: #e0e0e0;
    padding: 6px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background: #0f3460; color: white; }
"""
