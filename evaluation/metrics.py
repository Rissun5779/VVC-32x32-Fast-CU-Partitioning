import numpy as np

def compute_time_saving(t_anchor: float, t_proposed: float) -> float:
    """Calculate encoding time saving percentage (Delta T)."""
    return (t_anchor - t_proposed) / t_anchor * 100.0

def compute_bd_rate(r1, psnr1, r2, psnr2) -> float:
    """Bjontegaard Delta Rate (BD-Rate %) using 3rd-order polynomial interpolation."""
    lr1, lr2 = np.log10(r1), np.log10(r2)
    p1, p2 = np.array(psnr1), np.array(psnr2)
    min_int, max_int = max(p1.min(), p2.min()), min(p1.max(), p2.max())
    poly1, poly2 = np.polyfit(p1, lr1, 3), np.polyfit(p2, lr2, 3)
    int1 = np.polyval(np.polyint(poly1), max_int) - np.polyval(np.polyint(poly1), min_int)
    int2 = np.polyval(np.polyint(poly2), max_int) - np.polyval(np.polyint(poly2), min_int)
    avg_diff = (int2 - int1) / (max_int - min_int)
    return (10 ** avg_diff - 1.0) * 100.0
