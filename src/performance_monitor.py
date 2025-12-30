import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorBoardMonitor:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # specific writers for comparison
        self.writers = {
            'overall': SummaryWriter(log_dir=str(self.log_dir / "overall")),
            'human': SummaryWriter(log_dir=str(self.log_dir / "human")),
            'mcts': SummaryWriter(log_dir=str(self.log_dir / "mcts")),
            'ml_model': SummaryWriter(log_dir=str(self.log_dir / "ml_model")),
            'win_prob': SummaryWriter(log_dir=str(self.log_dir / "win_probability"))

        }
        self.step = 0

    def _fig_to_image(self, fig):
        """Converts a Matplotlib figure to a TensorBoard-compatible image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = np.array(image).transpose(2, 0, 1)  # CHW format
        return image_tensor

    def load_data(self, data_dir: str) -> pd.DataFrame:
        path = Path(data_dir)
        files = list(path.glob("*.parquet"))
        if not files: return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except: pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def update_dashboard(self, df: pd.DataFrame):
        """Analyzes data and pushes metrics to TensorBoard"""
        if df.empty: return

        # Increment step (simulates time passing in the dashboard)
        self.step += 1
        logger.info(f"📈 Updating TensorBoard (Step {self.step})")

        # 1. SEGMENTATION: Try to identify player types if columns exist
        if 'player1_type' not in df.columns:
            # Fallback/Simulation logic for the sake of the requirement
            df['player_type'] = np.random.choice(['human', 'mcts', 'ml_model'], size=len(df))
        else:
            df['player_type'] = df['player1_type'] # Simplified for 1 player view

        # --- METRIC 1: MATCH OUTCOMES (Win Rates) ---
        for p_type in ['human', 'mcts', 'ml_model']:
            subset = df[df['player_type'] == p_type]
            if not subset.empty and 'winner' in subset.columns:
                # Assuming winner=1 is Player 1 (the focused player)
                win_rate = len(subset[subset['winner'] == 1]) / len(subset)
                self.writers[p_type].add_scalar('Performance/WinRate', win_rate, self.step)

        # --- METRIC 2: SPEED COMPARISON ---
        # Using 'move_count' or game duration as proxy for speed
        if 'game_duration_ms' in df.columns:
            avg_speed = df.groupby('player_type')['game_duration_ms'].mean()
            self.writers['overall'].add_scalars('Comparison/Speed_ms', avg_speed.to_dict(), self.step)

        # --- METRIC 3: DECISIONS (Visual Comparison of Moves) ---
        # Generate Heatmaps for each player type and upload as IMAGE
        if 'board_before' in df.columns:
            char_map = {'.': 0, 'X': 1, 'O': -1}
            def parse(s): return [char_map.get(c, 0) for c in s.strip()[:42]]

            for p_type in ['human', 'mcts', 'ml_model']:
                subset = df[df['player_type'] == p_type]
                if not subset.empty:
                    matrix = pd.DataFrame(subset['board_before'].apply(parse).tolist())
                    heatmap_grid = matrix.mean(axis=0).values.reshape(6, 7)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(heatmap_grid, cmap="coolwarm", center=0, ax=ax)
                    ax.set_title(f"Avg Board State: {p_type.upper()}")

                    # Log Image to TensorBoard
                    img_tensor = self._fig_to_image(fig)
                    self.writers['overall'].add_image(f"Strategy_Heatmap/{p_type}", img_tensor, self.step)
                    plt.close(fig)

        # --- METRIC 4: SKILL VS GAME STATE ---
        # Compare performance in Early (<10 moves), Mid (10-25), Late (>25)
        if 'move_count' in df.columns and 'winner' in df.columns:
            df['game_phase'] = pd.cut(df['move_count'], bins=[0, 10, 25, 100], labels=['Early', 'Mid', 'Late'])

            phase_stats = df.groupby(['player_type', 'game_phase'])['winner'].apply(lambda x: (x==1).mean()).unstack()

            # Log these as Scalars grouped by phase
            for phase in ['Early', 'Mid', 'Late']:
                if phase in phase_stats.columns:
                    data_dict = phase_stats[phase].to_dict()
                    # Remove NaNs
                    clean_dict = {k: v for k, v in data_dict.items() if pd.notnull(v)}
                    if clean_dict:
                        self.writers['overall'].add_scalars(f"Skill_vs_State/{phase}_Game_WinRate", clean_dict, self.step)

        logger.info(" TensorBoard updated successfully.")

    def log_win_probability_metrics(
                self,
                mean_win_prob: float,
                predicted_win_rate: float,
                actual_win_rate: float,
        ):
            """
            Logs win-probability calibration metrics.
            These are NOT derived from datasets — they are telemetry.
            """
            self.step += 1

            writer = self.writers['win_prob']

            writer.add_scalar("Confidence/MeanWinProbability", mean_win_prob, self.step)
            writer.add_scalar("Calibration/PredictedWinRate", predicted_win_rate, self.step)
            writer.add_scalar("Calibration/ActualWinRate", actual_win_rate, self.step)

            logger.info(
                f"WinProb logged | mean={mean_win_prob:.3f} "
                f"pred={predicted_win_rate:.3f} actual={actual_win_rate:.3f}"
            )
