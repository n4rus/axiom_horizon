"""Plot training metrics from daemon cycles."""
import csv
import sys
from pathlib import Path
STATE = Path(__file__).parent / '.axiom_state'
CSV = STATE / 'training_metrics.csv'
if not CSV.exists():
    print(f'no data yet at {CSV}')
    sys.exit(0)
rows = list(csv.DictReader(open(CSV)))
if not rows:
    print('empty')
    sys.exit(0)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('install matplotlib: pip install matplotlib')
    sys.exit(1)
cycles = [int(r['cycle']) for r in rows]
vfes = [float(r['vfe']) for r in rows]
taus = [float(r['tau']) for r in rows]
vars_ = [float(r['variance']) for r in rows]
xis = [float(r['xi']) for r in rows]
times = [float(r['cycle_time_s']) for r in rows]
epochs = [float(r['epoch_age']) for r in rows]
_, axs = plt.subplots(3, 2, figsize=(14, 10))
axs[0,0].plot(cycles, vfes, alpha=0.7)
axs[0,0].set_title(f'VFE (last={vfes[-1]:.4f})')
axs[0,1].plot(cycles, taus, alpha=0.7)
axs[0,1].set_title(f'tau (last={taus[-1]:.4f})')
axs[1,0].plot(cycles, vars_, alpha=0.7)
axs[1,0].set_title(f'variance (last={vars_[-1]:.4f})')
axs[1,1].plot(cycles, xis, alpha=0.7)
axs[1,1].set_title(f'Xi (last={xis[-1]:.4f})')
axs[2,0].plot(cycles, times, alpha=0.7)
axs[2,0].set_title(f'cycle time (last={times[-1]:.1f}s)')
axs[2,1].plot(cycles, epochs, alpha=0.7)
axs[2,1].set_title(f'epoch age (last={epochs[-1]:.1f})')
plt.tight_layout()
plt.savefig(str(STATE / 'training_plot.png'))
print(f'plot saved, {len(rows)} cycles')
print(f'VFE range: {min(vfes):.4f}–{max(vfes):.4f} last={vfes[-1]:.4f}')
print(f'tau range: {min(taus):.4f}–{max(taus):.4f} last={taus[-1]:.4f}')
