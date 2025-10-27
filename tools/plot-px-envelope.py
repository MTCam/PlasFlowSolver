#!/usr/bin/env python3
# tools/plot-px-envelope.py
from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

Point = Tuple[float, float]  # (P_kPa, q_Wcm2)


# ---------- simple resolver for mechanism-like gas inputs -> facility names ----------
def _pick_canonical(target: str, available: List[str]) -> Optional[str]:
    t = target.lower()
    for name in available:
        if name.lower() == t:
            return name
    return None


def resolve_gas_name_simple(mech_name: str, available_names) -> Optional[str]:
    s = mech_name.strip().lower()
    if s.endswith(".xml"): s = s[:-4]
    if s.startswith("co2"):
        return _pick_canonical("CO2", list(available_names))
    if "air" in s:
        return _pick_canonical("Air", list(available_names))
    if s in {"nitrogen2", "nitrogen5", "test_n2", "n2"}:
        return _pick_canonical("N2", list(available_names))
    # last resort: case-insensitive exact match to any available name
    return _pick_canonical(s, list(available_names))


# ---------- geometry ----------
def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def convex_hull(points: List[Point]) -> List[Point]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_on_segment(p: Point, a: Point, b: Point, eps: float = 1e-12) -> bool:
    (x,y), (x1,y1), (x2,y2) = p, a, b
    area2 = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    if abs(area2) > eps*max(1.0, abs(x), abs(y), abs(x1), abs(y1), abs(x2), abs(y2)):
        return False
    return (min(x1,x2)-eps <= x <= max(x1,x2)+eps and
            min(y1,y2)-eps <= y <= max(y1,y2)+eps)


def point_in_polygon(point: Point, poly: List[Point], eps: float = 1e-12) -> bool:
    # inside or on boundary
    x, y = point
    n = len(poly)
    # boundary check
    for i in range(n):
        if point_on_segment(point, poly[i], poly[(i+1)%n], eps):
            return True
    inside = False
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if y1 > y2:
            x1,y1,x2,y2 = x2,y2,x1,y1
        if y < y1 or y >= y2:
            continue
        if y2 == y1:
            continue
        t = (y - y1) / (y2 - y1)
        xint = x1 + t*(x2 - x1)
        if xint > x + eps:
            inside = not inside
    return inside


# ---------- CSV ingest (facility schema: kPa & W/cm^2) ----------
def read_facility_csv(path: str) -> Dict[str, List[Point]]:
    with open(path, newline="") as f:
        # allow comment lines starting with '#'
        rows = [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#")]
    rdr = csv.DictReader(rows)
    req = {"plasma gas", "stagnation pressure [kPa]", "heat flux [W/cm^2]"}
    if not rdr.fieldnames or not req.issubset(set(rdr.fieldnames)):
        raise SystemExit("CSV must have columns: plasma gas, stagnation pressure [kPa], heat flux [W/cm^2]")
    by_gas: Dict[str, List[Point]] = defaultdict(list)
    for r in rdr:
        gas = (r["plasma gas"] or "").strip()
        if not gas:
            continue
        P = float(str(r["stagnation pressure [kPa]"]).replace(",", ""))
        q = float(str(r["heat flux [W/cm^2]"]).replace(",", ""))
        by_gas[gas].append((P, q))
    return by_gas


# ---------- plotting ----------
def plot_bounds(all_pts: Dict[str, List[Point]],
                gas_filter: Optional[List[str]],
                show_points: bool,
                fill_polys: bool,
                user: Optional[Tuple[str, float, float]],
                title: Optional[str],
                out: Optional[str]) -> None:
    gases = sorted(all_pts.keys())
    if gas_filter:
        gases = [g for g in gases if g in gas_filter]
        missing = [g for g in gas_filter if g not in all_pts]
        if missing:
            print(f"[plot] Note: requested gases not found in CSV: {missing}", file=sys.stderr)

    plt.figure(figsize=(8,6))
    hulls: Dict[str, List[Point]] = {}

    for gas in gases:
        pts = all_pts[gas]
        hull = convex_hull(pts) if len(pts) >= 3 else pts
        hulls[gas] = hull
        if len(hull) >= 3:
            hx = [p[0] for p in hull] + [hull[0][0]]
            hy = [p[1] for p in hull] + [hull[0][1]]
            line, = plt.plot(hx, hy, label=f"{gas} (polygon)")
            if fill_polys:
                col = line.get_color()
                plt.fill(hx, hy, facecolor=col, alpha=0.15, edgecolor=None)
        if show_points and pts:
            px = [p[0] for p in pts]; py = [p[1] for p in pts]
            plt.scatter(px, py, marker='x', label=f"{gas} (points)")

    # Optional user point
    if user:
        u_gas_raw, PkPa, qWcm2 = user
        resolved = resolve_gas_name_simple(u_gas_raw, gases)
        lbl = f"user: {u_gas_raw}"
        color = None
        if resolved and resolved in hulls and len(hulls[resolved]) >= 3:
            inside = point_in_polygon((PkPa, qWcm2), hulls[resolved])
            lbl = f"user: {resolved} ({'inside' if inside else 'outside'})"
            # try to match color of resolved gas outline
            # (retrieve color by plotting an invisible line with same label)
            for line in plt.gca().get_lines():
                if line.get_label().startswith(resolved):
                    color = line.get_color()
                    break
        plt.scatter([PkPa], [qWcm2], marker='*', s=150, edgecolor='k', facecolor=color, label=lbl, zorder=5)

    plt.xlabel("Stagnation Pressure [kPa]")
    plt.ylabel("Heat Flux [W/cm²]")
    if title: plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=200)
        print(f"[plot] Saved: {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot facility bounds polygons (kPa, W/cm²).")
    ap.add_argument("csv", help="Bounds CSV path (facility schema).")
    ap.add_argument("--gas", help="Comma-separated gas list, e.g. Air,N2 (default: all).")
    ap.add_argument("--out", help="Save figure to this file instead of showing.")
    ap.add_argument("--title", help="Figure title.")
    ap.add_argument("--no-points", action="store_true", help="Hide raw sample points.")
    ap.add_argument("--no-fill", action="store_true", help="Do not shade polygons (lines only).")
    ap.add_argument("--user-gas", help="User input gas/mechanism (e.g., air_11, nitrogen2, test_N2).")
    ap.add_argument("--user-PkPa", type=float, help="User stagnation pressure in kPa.")
    ap.add_argument("--user-qWcm2", type=float, help="User heat flux in W/cm².")
    args = ap.parse_args()

    all_pts = read_facility_csv(args.csv)
    gases = None
    if args.gas:
        gases = [g.strip() for g in args.gas.split(",") if g.strip()]

    user = None
    if args.user_gas is not None and args.user_PkPa is not None and args.user_qWcm2 is not None:
        user = (args.user_gas, args.user_PkPa, args.user_qWcm2)
    elif any(v is not None for v in (args.user_gas, args.user_PkPa, args.user_qWcm2)):
        print("[plot] Incomplete --user-* triple provided; ignoring user point.", file=sys.stderr)

    plot_bounds(
        all_pts,
        gas_filter=gases,
        show_points=not args.no_points,
        fill_polys=not args.no_fill,
        user=user,
        title=args.title or "Facility Bounds",
        out=args.out
    )


if __name__ == "__main__":
    main()
