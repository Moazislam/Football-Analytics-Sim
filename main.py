import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import poisson

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTableWidget,
                             QTableWidgetItem, QProgressBar, QHeaderView, 
                             QFrame, QMessageBox, QTabWidget, QSplitter, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QIcon

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Suppress pandas warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
LEAGUE_FILES = {
    "🇬🇧 Premier League": "E0.csv",
    "🇪🇸 La Liga": "SP1.csv",
    "🇩🇪 Bundesliga": "D1.csv",
    "🇮🇹 Serie A": "I1.csv",
    "🇫🇷 Ligue 1": "F1.csv"
}

SIMULATIONS = 10000
DECAY_FACTOR = 0.95 
HOME_ADVANTAGE = 1.1

# --- WORKER THREAD (SIMULATION LOGIC) ---
class SimulationWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.simulations = SIMULATIONS
        self.decay_factor = DECAY_FACTOR
        self.home_advantage = HOME_ADVANTAGE
        
    def run(self):
        try:
            results = self.run_optimized_simulation()
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

    def run_optimized_simulation(self):
        df = self.df.copy()
        current_season = df['Season'].max()
        
        current_teams = sorted(
            set(df.loc[df['Season'] == current_season, 'HomeTeam'])
            .union(df.loc[df['Season'] == current_season, 'AwayTeam'])
        )
        
        max_season = df['Season'].max()
        df['DecayWeight'] = self.decay_factor ** (max_season - df['Season'])
        
        played = df.dropna(subset=['FullTimeHomeGoals', 'FullTimeAwayGoals']).copy()
        attack, defense = self.estimate_team_strengths(played, current_teams)
        
        avg_attack = np.mean(list(attack.values()))
        for t in current_teams:
            attack[t] /= avg_attack
            defense[t] /= avg_attack

        played_current = played[played['Season'] == current_season]
        current_standings = self.calculate_current_standings(played_current, current_teams)
        future = self.get_remaining_fixtures(df, current_season, current_teams, played_current)
        lambda_home, lambda_away = self.calculate_expected_goals(future, attack, defense, current_teams)

        return self.monte_carlo_simulation(current_standings, current_teams, future, lambda_home, lambda_away)

    def estimate_team_strengths(self, played, current_teams):
        attack, defense = {}, {}
        has_xg = 'HomeTeamxG' in played.columns and 'AwayTeamxG' in played.columns
        
        for team in current_teams:
            home = played[played['HomeTeam'] == team]
            away = played[played['AwayTeam'] == team]
            
            if has_xg:
                h_gf = (home['HomeTeamxG'] * home['DecayWeight']).sum()
                a_gf = (away['AwayTeamxG'] * away['DecayWeight']).sum()
                h_ga = (home['AwayTeamxG'] * home['DecayWeight']).sum()
                a_ga = (away['HomeTeamxG'] * away['DecayWeight']).sum()
            else:
                h_gf = (home['FullTimeHomeGoals'] * home['DecayWeight']).sum()
                a_gf = (away['FullTimeAwayGoals'] * away['DecayWeight']).sum()
                h_ga = (home['FullTimeAwayGoals'] * home['DecayWeight']).sum()
                a_ga = (away['FullTimeHomeGoals'] * away['DecayWeight']).sum()

            weight = home['DecayWeight'].sum() + away['DecayWeight'].sum()
            if weight > 0:
                attack[team] = (h_gf + a_gf) / weight
                defense[team] = (h_ga + a_ga) / weight
            else:
                attack[team] = 1.0; defense[team] = 1.0
                
        return attack, defense

    def calculate_current_standings(self, played, teams):
        standings = {t: {'p': 0, 'gf': 0, 'ga': 0, 'pts': 0} for t in teams}
        for _, row in played.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            if h not in teams or a not in teams: continue
            
            hg, ag = int(row['FullTimeHomeGoals']), int(row['FullTimeAwayGoals'])
            standings[h]['p'] += 1; standings[a]['p'] += 1
            standings[h]['gf'] += hg; standings[a]['gf'] += ag
            standings[h]['ga'] += ag; standings[a]['ga'] += hg
            
            if hg > ag: standings[h]['pts'] += 3
            elif ag > hg: standings[a]['pts'] += 3
            else: standings[h]['pts'] += 1; standings[a]['pts'] += 1
        return standings

    def get_remaining_fixtures(self, df, season, teams, played):
        future = df[(df['Season'] == season) & (df['FullTimeHomeGoals'].isna())].copy()
        future = future[future['HomeTeam'].isin(teams) & future['AwayTeam'].isin(teams)]
        
        if len(future) == 0:
            played_pairs = set(zip(played['HomeTeam'], played['AwayTeam']))
            all_fixtures = [(h, a) for h in teams for a in teams if h != a]
            remaining = [f for f in all_fixtures if f not in played_pairs]
            future = pd.DataFrame(remaining, columns=['HomeTeam', 'AwayTeam'])
            
        return future

    def calculate_expected_goals(self, future, attack, defense, teams):
        h_teams = future['HomeTeam'].values
        a_teams = future['AwayTeam'].values
        att_h = np.array([attack[t] for t in h_teams])
        def_a = np.array([defense[t] for t in a_teams])
        att_a = np.array([attack[t] for t in a_teams])
        def_h = np.array([defense[t] for t in h_teams])
        
        league_avg = 1.35
        lambda_h = att_h * def_a * self.home_advantage * league_avg
        lambda_a = att_a * def_h * league_avg
        return lambda_h, lambda_a

    def monte_carlo_simulation(self, standings, teams, future, lam_h, lam_a):
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}
        
        final_pts = np.zeros((self.simulations, n_teams))
        final_gd = np.zeros((self.simulations, n_teams))
        final_gf = np.zeros((self.simulations, n_teams))
        
        base_pts = np.array([standings[t]['pts'] for t in teams])
        base_gf = np.array([standings[t]['gf'] for t in teams])
        base_ga = np.array([standings[t]['ga'] for t in teams])

        h_goals = poisson.rvs(lam_h, size=(self.simulations, len(future)))
        a_goals = poisson.rvs(lam_a, size=(self.simulations, len(future)))
        
        h_wins = h_goals > a_goals
        a_wins = a_goals > h_goals
        draws = h_goals == a_goals
        
        h_pts_added = h_wins * 3 + draws * 1
        a_pts_added = a_wins * 3 + draws * 1
        
        future_h_idx = [team_idx[t] for t in future['HomeTeam']]
        future_a_idx = [team_idx[t] for t in future['AwayTeam']]
        
        for sim in range(self.simulations):
            sim_pts = base_pts.copy()
            sim_gf = base_gf.copy()
            sim_ga = base_ga.copy()
            
            np.add.at(sim_pts, future_h_idx, h_pts_added[sim])
            np.add.at(sim_pts, future_a_idx, a_pts_added[sim])
            np.add.at(sim_gf, future_h_idx, h_goals[sim])
            np.add.at(sim_gf, future_a_idx, a_goals[sim])
            np.add.at(sim_ga, future_h_idx, a_goals[sim])
            np.add.at(sim_ga, future_a_idx, h_goals[sim])
            
            final_pts[sim] = sim_pts
            final_gf[sim] = sim_gf
            final_gd[sim] = sim_gf - sim_ga
            
            if (sim + 1) % 2000 == 0:
                self.progress.emit(int((sim + 1) / self.simulations * 100))

        results = []
        sort_keys = ( -final_gf, -final_gd, -final_pts )
        ranks = np.lexsort(sort_keys, axis=1)
        positions = np.zeros_like(ranks)
        np.put_along_axis(positions, ranks, np.repeat(np.arange(n_teams)[None, :], self.simulations, axis=0), axis=1)

        for i, team in enumerate(teams):
            team_ranks = positions[:, i]
            res = {
                'team': team,
                'current_played': standings[team]['p'],
                'current_points': standings[team]['pts'],
                'avg_points': round(np.mean(final_pts[:, i]), 1),
                'avg_gd': round(np.mean(final_gd[:, i]), 1),
                'win_prob': np.mean(team_ranks == 0) * 100,
                'top4_prob': np.mean(team_ranks < 4) * 100,
                'top6_prob': np.mean(team_ranks < 6) * 100,
                'relegation_prob': np.mean(team_ranks >= 17) * 100,
                'position_probs': [(np.sum(team_ranks == p) / self.simulations * 100) for p in range(n_teams)]
            }
            results.append(res)
            
        results.sort(key=lambda x: x['avg_points'], reverse=True)
        return {'results': results}


# --- MAIN GUI ---
class OptaSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize attributes to None to prevent crashes
        self.df = None
        self.results = None
        
        # Build UI first
        self.init_ui()
        
        # Now safely load the first league
        if LEAGUE_FILES:
            first_league = list(LEAGUE_FILES.keys())[0]
            # Explicitly call load (setCurrentText sometimes doesn't trigger on startup)
            self.load_league_data(first_league)
        
    def init_ui(self):
        self.setWindowTitle("Advanced Season Simulator & Stats Explorer")
        
        # Theme Styles
        self.setStyleSheet("""
            QMainWindow { background: #0f172a; }
            QLabel { color: #e2e8f0; }
            QTabWidget::pane { border: 1px solid #475569; border-radius: 4px; background: #1e293b; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 10px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #7c3aed; color: white; font-weight: bold; }
            QPushButton { background-color: #7c3aed; color: white; border-radius: 6px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #6d28d9; }
            QPushButton:disabled { background-color: #475569; color: #94a3b8; }
            QComboBox { background: #334155; color: white; padding: 5px; border-radius: 4px; border: 1px solid #475569; font-size: 14px; font-weight: bold; }
            QComboBox::drop-down { border: none; }
            QTableWidget { background-color: #1e293b; color: #e2e8f0; gridline-color: #334155; border: none; font-size: 13px; }
            QHeaderView::section { background-color: #0f172a; color: #94a3b8; padding: 8px; border: none; border-bottom: 2px solid #7c3aed; font-weight: bold; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Top Header ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background: #1e1b4b; border-radius: 12px; margin-bottom: 10px;")
        header_layout = QHBoxLayout(header_frame)
        
        title_box = QVBoxLayout()
        title = QLabel("Phenoix Opta: Analytics Suite")
        title.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #a78bfa;")
        subtitle = QLabel("Monte Carlo Simulation & Advanced Stats (ESC to Exit Full Screen)")
        subtitle.setStyleSheet("color: #94a3b8;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        
        # LEAGUE SELECTOR (Create but do not populate yet!)
        league_box = QVBoxLayout()
        league_label = QLabel("Active League:")
        league_label.setStyleSheet("color: #94a3b8; font-size: 11px; text-transform: uppercase;")
        self.combo_league = QComboBox()
        self.combo_league.setMinimumWidth(250)
        
        league_box.addWidget(league_label)
        league_box.addWidget(self.combo_league)
        header_layout.addLayout(league_box)
        
        main_layout.addWidget(header_frame)
        
        # Main Tab Widget (Creates all sub-widgets like tables/labels)
        self.main_tabs = QTabWidget()
        
        # TAB 1: SIMULATION
        self.sim_tab = self.create_simulation_tab()
        self.main_tabs.addTab(self.sim_tab, "🔮 Season Prediction")
        
        # TAB 2: STATS EXPLORER
        self.stats_tab = self.create_stats_tab()
        self.main_tabs.addTab(self.stats_tab, "📊 Stats Explorer")
        
        main_layout.addWidget(self.main_tabs)
        
        # --- SAFE INITIALIZATION ---
        # Only now, after all UI elements exist, do we populate the dropdown.
        # This prevents "AttributeError: 'OptaSimulator' object has no attribute 'lbl_info'"
        self.combo_league.addItems(LEAGUE_FILES.keys())
        self.combo_league.currentTextChanged.connect(self.load_league_data)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def load_league_data(self, league_name):
        filename = LEAGUE_FILES.get(league_name)
        if not filename: return
        
        self.df = None
        self.results = None
        
        # Reset UI elements
        # Because init_ui is done, we are guaranteed these exist now
        self.sim_table.setRowCount(0)
        self.heatmap_table.setRowCount(0)
        self.sim_figure.clear()
        self.sim_canvas.draw()
        self.stats_table.setRowCount(0)
        self.combo_season.clear()
        self.combo_metric.clear()
        
        if Path(filename).exists():
            try:
                self.df = pd.read_csv(filename)
                
                self.lbl_info.setText(f"Loaded: {league_name} ({len(self.df)} matches)")
                self.lbl_info.setStyleSheet("color: #22c55e;")
                self.btn_run.setEnabled(True)
                
                seasons = sorted(self.df['Season'].unique(), reverse=True)
                self.combo_season.addItem("All Seasons")
                for s in seasons:
                    self.combo_season.addItem(str(s))
                
                self.combo_metric.addItem("🏆 League Standings")
                numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
                ignore_cols = ['Season', 'DecayWeight']
                priority = ['HomeTeamxG', 'AwayTeamxG', 'FullTimeHomeGoals', 'FullTimeAwayGoals']
                for p in priority:
                    if p in numeric_cols: self.combo_metric.addItem(p)
                for c in numeric_cols:
                    if c not in priority and c not in ignore_cols: self.combo_metric.addItem(c)
                
                self.update_stats_view()
                
            except Exception as e:
                self.lbl_info.setText(f"Error: {e}")
                self.btn_run.setEnabled(False)
        else:
            self.lbl_info.setText(f"File not found: {filename}")
            self.lbl_info.setStyleSheet("color: #ef4444;")
            self.btn_run.setEnabled(False)

    # ---------------------------------------------------------
    # TAB 1: SIMULATION LOGIC
    # ---------------------------------------------------------
    def create_simulation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        control_frame = QFrame()
        control_frame.setStyleSheet("background: #1e293b; border-radius: 8px;")
        hbox = QHBoxLayout(control_frame)
        self.lbl_info = QLabel("Initializing...")
        self.btn_run = QPushButton("▶ Run Monte Carlo")
        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_run.setEnabled(False)
        hbox.addWidget(self.lbl_info)
        hbox.addStretch()
        hbox.addWidget(self.btn_run)
        layout.addWidget(control_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { text-align: center; border: 1px solid #475569; border-radius: 4px; height: 15px; } QProgressBar::chunk { background: #7c3aed; }")
        layout.addWidget(self.progress_bar)
        
        self.sim_inner_tabs = QTabWidget()
        self.sim_inner_tabs.addTab(self.create_sim_table(), "Standings Table")
        self.sim_inner_tabs.addTab(self.create_sim_viz(), "Heatmap & Charts")
        layout.addWidget(self.sim_inner_tabs)
        
        return widget

    def create_sim_table(self):
        self.sim_table = QTableWidget()
        self.sim_table.setColumnCount(10)
        self.sim_table.setHorizontalHeaderLabels(["Pos", "Team", "Played", "Pts", "Exp Pts", "Exp GD", "Title %", "Top 4 %", "Top 6 %", "Rel %"])
        self.sim_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.sim_table.setAlternatingRowColors(True)
        return self.sim_table

    def create_sim_viz(self):
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.heatmap_table = QTableWidget()
        self.heatmap_table.setColumnCount(21)
        self.heatmap_table.setHorizontalHeaderLabels(["TEAM"] + [str(i) for i in range(1, 21)])
        self.heatmap_table.verticalHeader().setVisible(False)
        self.heatmap_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 21): self.heatmap_table.setColumnWidth(i, 35)
        splitter.addWidget(self.heatmap_table)
        
        self.sim_figure = plt.figure(facecolor='#0f172a')
        self.sim_canvas = FigureCanvas(self.sim_figure)
        splitter.addWidget(self.sim_canvas)
        splitter.setSizes([300, 500])
        return splitter

    def run_simulation(self):
        if self.df is None: return
        
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.sim_table.setRowCount(0)
        
        self.worker = SimulationWorker(self.df)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_sim_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def on_sim_finished(self, data):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.results = data['results']
        self.populate_sim_table(self.results)
        self.update_heatmap(self.results)
        self.update_sim_charts(self.results)
        QMessageBox.information(self, "Done", f"Simulation Complete!")

    def populate_sim_table(self, results):
        self.sim_table.setRowCount(len(results))
        self.sim_table.setIconSize(QSize(24, 24))
        for i, r in enumerate(results):
            self.sim_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            team_item = QTableWidgetItem(r['team'])
            logo = Path(f"logos/{r['team']}.png")
            if logo.exists(): team_item.setIcon(QIcon(str(logo)))
            self.sim_table.setItem(i, 1, team_item)
            
            self.sim_table.setItem(i, 2, QTableWidgetItem(str(r['current_played'])))
            self.sim_table.setItem(i, 3, QTableWidgetItem(str(r['current_points'])))
            
            item = QTableWidgetItem(str(r['avg_points']))
            item.setForeground(QColor("#4ade80"))
            item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            self.sim_table.setItem(i, 4, item)
            
            gd_item = QTableWidgetItem(f"{r['avg_gd']:+.1f}")
            gd_item.setForeground(QColor("#f87171") if r['avg_gd'] < 0 else QColor("#4ade80"))
            self.sim_table.setItem(i, 5, gd_item)
            
            self.set_prob(i, 6, r['win_prob'], "#fbbf24")
            self.set_prob(i, 7, r['top4_prob'], "#60a5fa")
            self.set_prob(i, 8, r['top6_prob'], "#a78bfa")
            self.set_prob(i, 9, r['relegation_prob'], "#f87171")

    def set_prob(self, r, c, val, color):
        item = QTableWidgetItem(f"{val:.1f}%")
        if val > 10: 
            item.setForeground(QColor(color))
            item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.sim_table.setItem(r, c, item)

    def update_heatmap(self, results):
        self.heatmap_table.setRowCount(len(results))
        for r_idx, data in enumerate(results):
            team_item = QTableWidgetItem(data['team'])
            logo = Path(f"logos/{data['team']}.png")
            if logo.exists(): team_item.setIcon(QIcon(str(logo)))
            self.heatmap_table.setItem(r_idx, 0, team_item)
            
            for c_idx, prob in enumerate(data['position_probs']):
                if prob < 0.1: continue
                item = QTableWidgetItem(f"{prob:.0f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                alpha = min(prob / 30.0, 1.0) * 255
                bg = QColor(236, 72, 153, int(alpha)) 
                item.setBackground(bg)
                item.setForeground(Qt.GlobalColor.white if alpha > 100 else Qt.GlobalColor.gray)
                self.heatmap_table.setItem(r_idx, c_idx+1, item)

    def update_sim_charts(self, results):
        self.sim_figure.clear()
        ax = self.sim_figure.add_subplot(111)
        ax.set_facecolor('#0f172a')
        
        top10 = results[:10][::-1]
        teams = [x['team'] for x in top10]
        pts = [x['avg_points'] for x in top10]
        
        bars = ax.barh(teams, pts, color='#7c3aed')
        ax.set_title("Projected Points (Top 10)", color='white', pad=20)
        ax.tick_params(colors='#94a3b8')
        ax.spines['bottom'].set_color('#475569')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        self.sim_canvas.draw()

    # ---------------------------------------------------------
    # TAB 2: STATS EXPLORER LOGIC
    # ---------------------------------------------------------
    def create_stats_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        filter_frame = QFrame()
        filter_frame.setStyleSheet("background: #1e293b; border-radius: 8px; padding: 10px;")
        filter_layout = QHBoxLayout(filter_frame)
        
        filter_layout.addWidget(QLabel("Season:"))
        self.combo_season = QComboBox()
        self.combo_season.currentIndexChanged.connect(self.update_stats_view)
        filter_layout.addWidget(self.combo_season)
        
        filter_layout.addSpacing(20)
        
        filter_layout.addWidget(QLabel("Metric:"))
        self.combo_metric = QComboBox()
        self.combo_metric.currentIndexChanged.connect(self.update_stats_view)
        filter_layout.addWidget(self.combo_metric)
        
        filter_layout.addStretch()
        layout.addWidget(filter_frame)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.stats_table = QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.verticalHeader().setVisible(False)
        splitter.addWidget(self.stats_table)
        
        self.stats_figure = Figure(facecolor='#0f172a')
        self.stats_canvas = FigureCanvas(self.stats_figure)
        splitter.addWidget(self.stats_canvas)
        
        splitter.setSizes([500, 300])
        layout.addWidget(splitter)
        
        return widget

    def update_stats_view(self):
        if self.df is None or self.combo_metric.count() == 0:
            return
            
        season_txt = self.combo_season.currentText()
        metric = self.combo_metric.currentText()
        
        if season_txt == "All Seasons":
            data = self.df.copy()
        else:
            try:
                s_int = int(season_txt)
                data = self.df[self.df['Season'] == s_int].copy()
            except:
                return

        self.update_stats_charts(data, metric)

        if metric == "🏆 League Standings":
            self.display_standings(data)
        else:
            self.display_generic_metric(data, metric)

    def update_stats_charts(self, data, metric):
        self.stats_figure.clear()
        
        text_color = '#94a3b8'
        bar_color = '#7c3aed'
        hist_color = '#38bdf8'
        bg_color = '#0f172a'

        if metric == "🏆 League Standings":
            teams = {}
            for _, row in data.iterrows():
                if pd.isna(row['FullTimeHomeGoals']): continue
                h, a = row['HomeTeam'], row['AwayTeam']
                hg, ag = row['FullTimeHomeGoals'], row['FullTimeAwayGoals']
                
                teams.setdefault(h, 0)
                teams.setdefault(a, 0)
                
                if hg > ag: teams[h] += 3
                elif ag > hg: teams[a] += 3
                else:
                    teams[h] += 1; teams[a] += 1
            
            sorted_teams = sorted(teams.items(), key=lambda item: item[1], reverse=False)
            names = [x[0] for x in sorted_teams[-10:]]
            values = [x[1] for x in sorted_teams[-10:]]
            
            ax = self.stats_figure.add_subplot(111)
            ax.barh(names, values, color=bar_color)
            ax.set_title("Current Points Leaderboard (Top 10)", color='white')
            ax.set_xlabel("Points", color=text_color)
            
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            ax.spines['bottom'].set_color('#475569')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            
        else:
            gs = self.stats_figure.add_gridspec(1, 2)
            ax1 = self.stats_figure.add_subplot(gs[0, 0])
            ax2 = self.stats_figure.add_subplot(gs[0, 1])

            group_col = "AwayTeam" if "Away" in metric else "HomeTeam"
            
            agg = data.groupby(group_col)[metric].sum().sort_values(ascending=True)
            top_agg = agg.tail(10)
            
            ax1.barh(top_agg.index, top_agg.values, color=bar_color)
            ax1.set_title(f"Total {metric} by Team (Top 10)", color='white', fontsize=10)
            
            ax2.hist(data[metric].dropna(), bins=15, color=hist_color, edgecolor='#0f172a', alpha=0.8)
            ax2.set_title(f"Distribution of {metric} (Per Match)", color='white', fontsize=10)
            ax2.set_ylabel("Frequency", color=text_color)

            for ax in [ax1, ax2]:
                ax.set_facecolor(bg_color)
                ax.tick_params(colors=text_color)
                ax.spines['bottom'].set_color('#475569')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#475569')

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def display_standings(self, data):
        columns = ["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]
        self.stats_table.setColumnCount(len(columns))
        self.stats_table.setHorizontalHeaderLabels(columns)
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        teams = {} 
        for _, row in data.iterrows():
            if pd.isna(row['FullTimeHomeGoals']) or pd.isna(row['FullTimeAwayGoals']):
                continue
                
            h, a = row['HomeTeam'], row['AwayTeam']
            hg, ag = int(row['FullTimeHomeGoals']), int(row['FullTimeAwayGoals'])
            
            if h not in teams: teams[h] = {'p':0, 'w':0, 'd':0, 'l':0, 'gf':0, 'ga':0, 'pts':0}
            if a not in teams: teams[a] = {'p':0, 'w':0, 'd':0, 'l':0, 'gf':0, 'ga':0, 'pts':0}
            
            teams[h]['p'] += 1; teams[h]['gf'] += hg; teams[h]['ga'] += ag
            teams[a]['p'] += 1; teams[a]['gf'] += ag; teams[a]['ga'] += hg
            
            if hg > ag:
                teams[h]['w'] += 1; teams[h]['pts'] += 3; teams[a]['l'] += 1
            elif ag > hg:
                teams[a]['w'] += 1; teams[a]['pts'] += 3; teams[h]['l'] += 1
            else:
                teams[h]['d'] += 1; teams[h]['pts'] += 1; teams[a]['d'] += 1; teams[a]['pts'] += 1
        
        standings_list = []
        for name, stats in teams.items():
            stats['team'] = name
            stats['gd'] = stats['gf'] - stats['ga']
            standings_list.append(stats)
            
        standings_list.sort(key=lambda x: (x['pts'], x['gd'], x['gf']), reverse=True)
        
        self.stats_table.setRowCount(len(standings_list))
        self.stats_table.setIconSize(QSize(24, 24))
        
        for i, s in enumerate(standings_list):
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            team_item = QTableWidgetItem(s['team'])
            logo = Path(f"logos/{s['team']}.png")
            if logo.exists(): team_item.setIcon(QIcon(str(logo)))
            self.stats_table.setItem(i, 1, team_item)
            
            self.stats_table.setItem(i, 2, QTableWidgetItem(str(s['p'])))
            self.stats_table.setItem(i, 3, QTableWidgetItem(str(s['w'])))
            self.stats_table.setItem(i, 4, QTableWidgetItem(str(s['d'])))
            self.stats_table.setItem(i, 5, QTableWidgetItem(str(s['l'])))
            self.stats_table.setItem(i, 6, QTableWidgetItem(str(s['gf'])))
            self.stats_table.setItem(i, 7, QTableWidgetItem(str(s['ga'])))
            
            gd_item = QTableWidgetItem(f"{s['gd']:+d}")
            if s['gd'] > 0: gd_item.setForeground(QColor("#4ade80"))
            elif s['gd'] < 0: gd_item.setForeground(QColor("#f87171"))
            self.stats_table.setItem(i, 8, gd_item)
            
            pts_item = QTableWidgetItem(str(s['pts']))
            pts_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            pts_item.setForeground(QColor("#a78bfa"))
            self.stats_table.setItem(i, 9, pts_item)

    def display_generic_metric(self, data, metric):
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Rank", "Team", "Total", "Per Game"])
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        if "Away" in metric: group_col = "AwayTeam"
        else: group_col = "HomeTeam"
            
        stats = data.groupby(group_col)[metric].agg(['sum', 'count', 'mean']).reset_index()
        stats.columns = ['Team', 'Total', 'Games', 'Average']
        stats = stats.sort_values(by='Total', ascending=False)
        
        self.stats_table.setRowCount(len(stats))
        self.stats_table.setIconSize(QSize(24, 24))
        
        for i, row in stats.iterrows():
            r_idx = stats.index.get_loc(i)
            self.stats_table.setItem(r_idx, 0, QTableWidgetItem(str(r_idx + 1)))
            
            team_item = QTableWidgetItem(row['Team'])
            logo = Path(f"logos/{row['Team']}.png")
            if logo.exists(): team_item.setIcon(QIcon(str(logo)))
            self.stats_table.setItem(r_idx, 1, team_item)
            
            self.stats_table.setItem(r_idx, 2, QTableWidgetItem(f"{row['Total']:.2f}"))
            self.stats_table.setItem(r_idx, 3, QTableWidgetItem(f"{row['Average']:.2f}"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptaSimulator()
    window.showFullScreen()
    sys.exit(app.exec())
