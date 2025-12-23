"""
Premier League Prediction System - Main Application
GUI demo mode (no R required)
"""

import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
    QTabWidget, QProgressBar, QTextEdit, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ======================================================
# 🔧 DEMO MODE FLAG
# ======================================================
DEMO_MODE = True   # <- switch to False when R backend is ready


# ======================================================
# 🔄 Background Analysis Thread
# ======================================================
class AnalysisThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, data_path, r_script_path):
        super().__init__()
        self.data_path = data_path
        self.r_script_path = r_script_path

    def run(self):
        try:
            if DEMO_MODE:
                self.run_demo_analysis()
            else:
                self.run_real_analysis()
        except Exception as e:
            self.error.emit(str(e))

    def run_demo_analysis(self):
        """Fake analysis to test GUI without R"""
        time.sleep(2)

        teams = [
            "Man City", "Arsenal", "Liverpool", "Chelsea", "Man United",
            "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham",
            "Brentford", "Crystal Palace", "Wolves", "Fulham",
            "Everton", "Bournemouth", "Forest", "Luton", "Burnley", "Sheffield Utd"
        ]

        league_table = pd.DataFrame({
            "Position": range(1, 21),
            "Team": teams,
            "Played": np.random.randint(24, 29, 20),
            "GoalDiff": np.random.randint(-25, 45, 20),
            "Points": np.sort(np.random.randint(25, 75, 20))[::-1]
        })

        predictions = pd.DataFrame({
            "Team": teams,
            "TitleProbability": np.round(
                np.random.dirichlet(np.ones(20)) * 100, 2
            ),
            "ExpectedPoints": np.random.uniform(45, 90, 20),
            "AvgPosition": np.random.uniform(1, 20, 20)
        })

        self.finished.emit({
            "league_table": league_table,
            "predictions": predictions
        })

    def run_real_analysis(self):
        """Real R analysis (disabled in demo mode)"""
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        ro.r.source(self.r_script_path)
        analyze_func = ro.globalenv["analyze_premier_league"]
        result = analyze_func(self.data_path)

        predictions = pandas2ri.rpy2py(result.rx2("predictions"))
        league_table = pandas2ri.rpy2py(result.rx2("league_table"))

        self.finished.emit({
            "league_table": league_table,
            "predictions": predictions
        })


# ======================================================
# 🏟️ Main Application
# ======================================================
class PremierLeagueApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.data_path = None
        self.league_table = None
        self.predictions = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Premier League Prediction System")
        self.setGeometry(100, 100, 1200, 800)
        self.set_pl_style()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.create_header())

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.create_home_tab()
        self.create_league_table_tab()
        self.create_predictions_tab()
        self.create_about_tab()

    def set_pl_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f8f8; }
            QLabel { color: #37003c; }
            QPushButton {
                background-color: #37003c;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4a0052; }
            QHeaderView::section {
                background-color: #37003c;
                color: white;
                padding: 1px;
                font-weight: bold;
            }
        """)

    def create_header(self):
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet("background-color: #37003c;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("PREMIER LEAGUE")
        title.setStyleSheet("color:white;font-size:28px;font-weight:bold;")

        subtitle = QLabel("Prediction Engine")
        subtitle.setStyleSheet("color:#00ff85;font-size:14px;font-weight:bold;")

        text_layout = QVBoxLayout()
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)

        layout.addLayout(text_layout)
        layout.addStretch()

        if DEMO_MODE:
            badge = QLabel("DEMO MODE")
            badge.setStyleSheet("""
                background:#ffcc00;
                color:#37003c;
                padding:6px 12px;
                font-weight:bold;
                border-radius:4px;
            """)
            layout.addWidget(badge)

        return header

    def create_home_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(40, 40, 40, 40)

        self.status_label = QLabel("No data loaded")
        load_btn = QPushButton("📁 Load Match Data CSV")
        load_btn.clicked.connect(self.load_data)

        self.analyze_btn = QPushButton("⚡ Run Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.run_analysis)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout.addWidget(load_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.progress)
        layout.addStretch()

        self.tabs.addTab(tab, "🏠 Home")

    def create_league_table_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.league_table_widget = QTableWidget()
        self.league_table_widget.setColumnCount(5)
        self.league_table_widget.setHorizontalHeaderLabels(
            ["Pos", "Team", "Played", "GD", "Points"]
        )
        self.league_table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.league_table_widget)
        self.tabs.addTab(tab, "📊 League Table")

    def create_predictions_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.predictions_table = QTableWidget()
        self.predictions_table.setColumnCount(4)
        self.predictions_table.setHorizontalHeaderLabels(
            ["Team", "Title %", "Exp Points", "Avg Pos"]
        )
        self.predictions_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.predictions_table)
        self.tabs.addTab(tab, "🎯 Predictions")

    def create_about_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Model Overview</h2>
        <p>Poisson regression with Monte Carlo simulation.</p>
        <p>This demo mode generates synthetic results to test the GUI.</p>
        """)
        layout.addWidget(text)
        self.tabs.addTab(tab, "ℹ️ About")

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv)")
        if path:
            self.data_path = path
            self.status_label.setText(f"Loaded: {path}")
            self.analyze_btn.setEnabled(True)

    def run_analysis(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.analyze_btn.setEnabled(False)

        self.thread = AnalysisThread(self.data_path, "analysis.R")
        self.thread.finished.connect(self.on_done)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_done(self, result):
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)

        self.league_table = result["league_table"]
        self.predictions = result["predictions"]

        self.populate_league_table()
        self.populate_predictions()

        QMessageBox.information(self, "Done", "Analysis completed")
        self.tabs.setCurrentIndex(2)

    def on_error(self, msg):
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def populate_league_table(self):
        self.league_table_widget.setRowCount(len(self.league_table))
        for i, row in self.league_table.iterrows():
            self.league_table_widget.setItem(i, 0, QTableWidgetItem(str(row["Position"])))
            self.league_table_widget.setItem(i, 1, QTableWidgetItem(row["Team"]))
            self.league_table_widget.setItem(i, 2, QTableWidgetItem(str(row["Played"])))
            self.league_table_widget.setItem(i, 3, QTableWidgetItem(str(row["GoalDiff"])))
            self.league_table_widget.setItem(i, 4, QTableWidgetItem(str(row["Points"])))

    def populate_predictions(self):
        self.predictions_table.setRowCount(len(self.predictions))
        for i, row in self.predictions.iterrows():
            self.predictions_table.setItem(i, 0, QTableWidgetItem(row["Team"]))
            self.predictions_table.setItem(i, 1, QTableWidgetItem(f"{row['TitleProbability']}%"))
            self.predictions_table.setItem(i, 2, QTableWidgetItem(f"{row['ExpectedPoints']:.1f}"))
            self.predictions_table.setItem(i, 3, QTableWidgetItem(f"{row['AvgPosition']:.1f}"))


# ======================================================
# 🚀 Entry Point
# ======================================================
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    window = PremierLeagueApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
