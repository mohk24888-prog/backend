from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from src.analytics.pass_detection import PassEvent
from src.core.pitch import SoccerPitchConfiguration
from src.io.generate_player_heatmap import draw_pitch, world_to_canvas

# BGR colors per team
_NODE_COLORS = {0: (147, 20, 255), 1: (255, 191, 0)}
_EDGE_COLORS = {0: (100, 14, 180), 1: (180, 134, 0)}


def generate_pass_network(
    passes: list[PassEvent],
    accum_positions: dict[int, list[tuple[int, float, float]]],
    accum_metadata: dict[int, tuple[str, int | None]],
    pitch_cfg: SoccerPitchConfiguration,
    output_dir: Path,
    *,
    width_px: int = 1000,
    height_px: int = 648,
    min_node_samples: int = 5,
) -> int:
    """Render per-team pass network PNGs and return the number of images created."""
    pitch_length = float(pitch_cfg.length)
    pitch_width = float(pitch_cfg.width)

    # Compute average pitch position per track.
    avg_pos: dict[int, tuple[float, float]] = {}
    for track_id, samples in accum_positions.items():
        if len(samples) < min_node_samples:
            continue
        xs = [s[1] for s in samples]
        ys = [s[2] for s in samples]
        avg_pos[track_id] = (sum(xs) / len(xs), sum(ys) / len(ys))

    # Build undirected edge counts per team.
    team_edges: dict[int, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for p in passes:
        if p.passer_id not in avg_pos or p.receiver_id not in avg_pos:
            continue
        key = (min(p.passer_id, p.receiver_id), max(p.passer_id, p.receiver_id))
        team_edges[p.team_id][key] += 1

    # Collect nodes per team.
    team_nodes: dict[int, set[int]] = defaultdict(set)
    for p in passes:
        if p.passer_id in avg_pos:
            team_nodes[p.team_id].add(p.passer_id)
        if p.receiver_id in avg_pos:
            team_nodes[p.team_id].add(p.receiver_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_created = 0

    for team_id in sorted(team_edges.keys()):
        edges = team_edges[team_id]
        nodes = team_nodes.get(team_id, set())
        if not edges:
            continue

        canvas = draw_pitch(width_px, height_px, pitch_cfg)

        # Title bar.
        title_h = 36
        cv2.rectangle(canvas, (0, 0), (width_px, title_h), (0, 0, 0), -1)
        title = f"Pass Network — Team {team_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        tsz = cv2.getTextSize(title, font, 0.7, 2)[0]
        cv2.putText(
            canvas,
            title,
            (width_px // 2 - tsz[0] // 2, title_h // 2 + tsz[1] // 2),
            font,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        edge_color = _EDGE_COLORS.get(team_id, (150, 150, 150))
        node_color = _NODE_COLORS.get(team_id, (200, 200, 200))

        # Draw edges.
        for (a, b), count in edges.items():
            if a not in avg_pos or b not in avg_pos:
                continue
            pt_a = world_to_canvas(
                avg_pos[a][0],
                avg_pos[a][1],
                pitch_length,
                pitch_width,
                width_px,
                height_px,
            )
            pt_b = world_to_canvas(
                avg_pos[b][0],
                avg_pos[b][1],
                pitch_length,
                pitch_width,
                width_px,
                height_px,
            )
            thickness = min(1 + count // 2, 8)
            cv2.line(canvas, pt_a, pt_b, edge_color, thickness, lineType=cv2.LINE_AA)

        # Draw nodes.
        for track_id in nodes:
            if track_id not in avg_pos:
                continue
            pt = world_to_canvas(
                avg_pos[track_id][0],
                avg_pos[track_id][1],
                pitch_length,
                pitch_width,
                width_px,
                height_px,
            )
            cv2.circle(canvas, pt, 10, node_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pt, 10, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            label = str(track_id)
            lsz = cv2.getTextSize(label, font, 0.4, 1)[0]
            cv2.putText(
                canvas,
                label,
                (pt[0] - lsz[0] // 2, pt[1] + lsz[1] // 2),
                font,
                0.4,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        out_path = output_dir / f"pass_network_team{team_id}.png"
        cv2.imwrite(str(out_path), canvas)
        images_created += 1

    return images_created
