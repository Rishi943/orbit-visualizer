"""
Orbit Visualizer — Beginner-Friendly, Hand-Coded Kepler Orbits
Author: Rishi
Description:
- Simulates simple 2D Keplerian orbits for multiple bodies around the Sun (focus).
- Saves static plots (full view + inner zoom) with a black background.
- Optionally saves a lightweight GIF animation (if 'imageio' is installed).
- Exports ephemerides (date, x, y, r, true anomaly) to Excel.

Requirements:
- Python 3.9+
- numpy, pandas, matplotlib, openpyxl (for Excel)
- Optional: imageio (for GIF). If missing, the script will skip the GIF step.

Design goals:
- Simple, readable, and well-commented.
- No online APIs or advanced dependencies.
- Non-interactive (matplotlib Agg) and Windows-friendly.
"""

import os
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt

# Try optional imageio for GIF export (graceful fallback if not installed)
try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

# ----------------------------
# Configuration (easy to tweak)
# ----------------------------
OUTPUT_DIR = "."
START_DATE = datetime(2025, 1, 1)    # Simulation start date
DAYS = 365                           # How many days to simulate
STEP_DAYS = 2                        # Time step in days (bigger = faster/leaner)
PLOT_SIZE = (8, 8)                   # Inches
INNER_VIEW_LIMIT = 2.5               # +/- AU for inner zoom plot (Mercury..Mars scale)
FULL_VIEW_LIMIT = 35                 # +/- AU for full plot (shows comet/aphelion nicely)
TRAIL_FRACTION = 0.30                # Fraction of recent points to show as "trail" in animation

# Bodies:
# a: semi-major axis (AU), e: eccentricity, period_days, color, label
# Periods are approximate (sufficient for a visualizer).
BODIES = [
    {"name": "Mercury", "a": 0.387, "e": 0.205, "period_days": 87.969,  "color": "#B0AFAF"},
    {"name": "Venus",   "a": 0.723, "e": 0.007, "period_days": 224.701, "color": "#C9B089"},
    {"name": "Earth",   "a": 1.000, "e": 0.017, "period_days": 365.256, "color": "#2E86C1"},  # blue/green vibe
    {"name": "Mars",    "a": 1.524, "e": 0.093, "period_days": 686.980, "color": "#C1440E"},
    {"name": "Jupiter", "a": 5.204, "e": 0.049, "period_days": 4332.589,"color": "#D4AF37"},
    {"name": "Saturn",  "a": 9.583, "e": 0.057, "period_days": 10759.22,"color": "#E5C07B"},
    # A comet with high eccentricity to look dramatic:
    {"name": "Comet",   "a": 10.0,  "e": 0.8,   "period_days": 3650.0,  "color": "#FFFFFF"},   # white comet
]

# Sun marker style (yellow on black background)
SUN_COLOR = "#FFD700"
SUN_SIZE = 150


# -----------------------------------
# Kepler utilities (elliptical orbits)
# -----------------------------------
def solve_kepler_equation(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation for elliptical orbits: E - e*sin(E) = M
    Using Newton-Raphson. Returns eccentric anomaly E in radians.
    M can be any real; we reduce to [-pi, pi] for stability.
    """
    # Normalize M to [-pi, pi] for numerical stability
    M = (M + math.pi) % (2 * math.pi) - math.pi

    # Initial guess: for small e, E ~ M; for larger e, a better guess helps
    E = M if e < 0.8 else math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def state_from_kepler(a: float, e: float, M: float) -> tuple[float, float, float, float]:
    """
    Given semi-major axis a (AU), eccentricity e, and mean anomaly M (rad),
    return (x, y, r, nu) in AU and radians for 2D orbit around Sun at focus.
      - Solve E from Kepler's eq
      - Convert to true anomaly nu
      - Radius r = a * (1 - e*cos E)
      - Position in orbital plane: x = r cos(nu), y = r sin(nu)
    """
    E = solve_kepler_equation(M, e)
    # True anomaly (nu) from eccentric anomaly (E):
    # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    sin_nu = (math.sqrt(1 - e**2) * math.sin(E)) / (1 - e * math.cos(E))
    cos_nu = (math.cos(E) - e) / (1 - e * math.cos(E))
    nu = math.atan2(sin_nu, cos_nu)

    r = a * (1 - e * math.cos(E))
    x = r * math.cos(nu)
    y = r * math.sin(nu)
    return x, y, r, nu


# -------------------------
# Ephemeris for all bodies
# -------------------------
def build_ephemeris(start_date: datetime,
                    days: int,
                    step_days: int,
                    bodies: list[dict]) -> pd.DataFrame:
    """
    Build long-form ephemeris DataFrame:
      columns = [date, body, x_au, y_au, r_au, true_anomaly_deg]
    """
    dates = [start_date + timedelta(days=int(d)) for d in range(0, days + 1, step_days)]

    rows = []
    for d_idx, date in enumerate(dates):
        for body in bodies:
            a = body["a"]
            e = body["e"]
            T = body["period_days"]
            # Mean motion n = 2π / T
            n = 2 * math.pi / T
            M = n * (d_idx * step_days)  # Mean anomaly from t=0; M0=0 for simplicity
            x, y, r, nu = state_from_kepler(a, e, M)
            rows.append({
                "date": date,
                "body": body["name"],
                "x_au": x,
                "y_au": y,
                "r_au": r,
                "true_anomaly_deg": math.degrees(nu)
            })

    df = pd.DataFrame(rows)
    return df


# -------------
# Plot helpers
# -------------
def style_axes(ax, title: str, limit_au: float):
    ax.set_title(title, color="w", pad=12)
    ax.set_xlabel("X (AU)", color="w")
    ax.set_ylabel("Y (AU)", color="w")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit_au, limit_au)
    ax.set_ylim(-limit_au, limit_au)
    ax.grid(True, color="#333333", alpha=0.35)
    # Make axes spines & ticks visible on dark
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.tick_params(colors="#DDDDDD")


def plot_static_views(df: pd.DataFrame,
                      bodies: list[dict],
                      full_path: str,
                      inner_path: str):
    plt.style.use("default")  # keep clean defaults
    # FULL VIEW
    fig, ax = plt.subplots(figsize=PLOT_SIZE, dpi=150)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Draw all points per body
    for body in bodies:
        sub = df[df["body"] == body["name"]]
        ax.plot(sub["x_au"], sub["y_au"], lw=1.0, color=body["color"], label=body["name"])

    # Sun
    ax.scatter([0], [0], s=SUN_SIZE, c=SUN_COLOR, edgecolors="none", zorder=5, label="Sun")
    style_axes(ax, "Orbit Visualizer — Full View", FULL_VIEW_LIMIT)
    ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#DDDDDD")
    plt.tight_layout()
    plt.savefig(full_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    # INNER VIEW (zoomed)
    fig2, ax2 = plt.subplots(figsize=PLOT_SIZE, dpi=150)
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")
    for body in bodies:
        sub = df[df["body"] == body["name"]]
        ax2.plot(sub["x_au"], sub["y_au"], lw=1.0, color=body["color"], label=body["name"])

    ax2.scatter([0], [0], s=SUN_SIZE, c=SUN_COLOR, edgecolors="none", zorder=5, label="Sun")
    style_axes(ax2, "Orbit Visualizer — Inner Planets", INNER_VIEW_LIMIT)
    ax2.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#DDDDDD")
    plt.tight_layout()
    plt.savefig(inner_path, facecolor=fig2.get_facecolor())
    plt.close(fig2)


def save_animation_gif(df: pd.DataFrame,
                       bodies: list[dict],
                       gif_path: str,
                       limit_au: float,
                       step_frames: int = 1):
    """
    Simple point-by-point GIF: advances one time step per frame.
    Uses a short "trail" for each body for a nicer look.
    """
    if not IMAGEIO_AVAILABLE:
        print("[GIF] imageio not installed — skipping animation.")
        return

    print("[GIF] Rendering frames (this is quick for small datasets)...")
    frames = []
    fig, ax = plt.subplots(figsize=PLOT_SIZE, dpi=140)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    style_axes(ax, "Orbit Visualizer — Animation", limit_au)

    # Pre-split body data to speed up frame loop
    series = {b["name"]: df[df["body"] == b["name"]].reset_index(drop=True) for b in bodies}
    max_len = max(len(s) for s in series.values())
    trail = max(1, int(TRAIL_FRACTION * max_len))

    for i in range(0, max_len, step_frames):
        ax.clear()
        ax.set_facecolor("black")
        style_axes(ax, "Orbit Visualizer — Animation", limit_au)
        ax.scatter([0], [0], s=SUN_SIZE, c=SUN_COLOR, edgecolors="none", zorder=5)

        for body in bodies:
            sub = series[body["name"]]
            j0 = max(0, i - trail)
            # Trail
            ax.plot(sub.loc[j0:i, "x_au"], sub.loc[j0:i, "y_au"], lw=1.2, color=body["color"], alpha=0.9)
            # Current marker (slightly larger)
            if i < len(sub):
                ax.scatter(sub.loc[i, "x_au"], sub.loc[i, "y_au"], c=body["color"], s=12, zorder=6)

        plt.tight_layout()
        # Convert current canvas to an image array for the GIF
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    imageio.mimsave(gif_path, frames, duration=0.06)  # ~16 fps
    plt.close(fig)
    print(f"[GIF] Saved: {gif_path}")


# ---------------
# Main entrypoint
# ---------------
def main():
    print("=== Orbit Visualizer (2D, Kepler Ellipses) ===")
    print(f"Simulation start: {START_DATE.date()}  | Days: {DAYS}  | Step: {STEP_DAYS} day(s)")
    print(f"Bodies: {[b['name'] for b in BODIES]}")

    # Build ephemeris
    df = build_ephemeris(START_DATE, DAYS, STEP_DAYS, BODIES)

    # Save Excel (one tidy sheet)
    excel_path = os.path.join(OUTPUT_DIR, "orbit_ephemeris.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"[Excel] Saved: {excel_path}  (rows: {len(df):,})")

    # Save static plots
    full_png = os.path.join(OUTPUT_DIR, "orbit_full_view.png")
    inner_png = os.path.join(OUTPUT_DIR, "orbit_inner_view.png")
    plot_static_views(df, BODIES, full_png, inner_png)
    print(f"[Plot] Saved: {full_png}")
    print(f"[Plot] Saved: {inner_png}")

    # Optional GIF animation (graceful skip if imageio missing)
    gif_path = os.path.join(OUTPUT_DIR, "orbit_animation.gif")
    save_animation_gif(df, BODIES, gif_path, limit_au=INNER_VIEW_LIMIT, step_frames=1)

    print("\nSummary:")
    for body in BODIES:
        sub = df[df["body"] == body["name"]]
        r_min = sub["r_au"].min()
        r_max = sub["r_au"].max()
        print(f" - {body['name']:<8} a={body['a']:.3f} AU  e={body['e']:.3f}  "
              f"r_min={r_min:.3f} AU  r_max={r_max:.3f} AU")
    print("\nProject complete! ✅ Ready for GitHub upload.")
    print("Files:\n - orbit_full_view.png\n - orbit_inner_view.png\n - orbit_ephemeris.xlsx\n - orbit_animation.gif (if imageio present)")


if __name__ == "__main__":
    main()
