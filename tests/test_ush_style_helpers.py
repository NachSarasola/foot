import sys
from pathlib import Path
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))
from ush_style import text_halo, avoid_overlap, edge_curved, save_fig_pro, COLORS


def test_text_halo_renders_box():
    fig, ax = plt.subplots()
    txt = text_halo(ax, "Hola", x=0.5, y=0.5, color=COLORS["ink"])
    bbox = txt.get_bbox_patch()
    assert bbox is not None
    fc = bbox.get_facecolor()
    assert all(abs(c - 1.0) < 1e-6 for c in fc[:3])
    assert fc[3] < 1.0
    plt.close(fig)


def test_avoid_overlap_separates_labels():
    fig, ax = plt.subplots()
    t1 = ax.text(0.5, 0.5, "A")
    t2 = ax.text(0.5, 0.5, "B")
    fig.canvas.draw()
    assert t1.get_window_extent().overlaps(t2.get_window_extent())
    avoid_overlap([t1, t2], padding=5)
    fig.canvas.draw()
    assert not t1.get_window_extent().overlaps(t2.get_window_extent())
    plt.close(fig)


def test_edge_curved_adds_shadow():
    fig, ax = plt.subplots()
    edge_curved(ax, (0, 0), (1, 1), weight=2, shadow=True)
    assert len(ax.patches) == 2
    for p in ax.patches:
        assert p.get_capstyle() == "round"
    plt.close(fig)


def test_save_fig_pro_creates_thumbnail(tmp_path):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "figure.png"
    save_fig_pro(fig, out)
    thumb = out.with_name("figure_thumb.png")
    assert out.exists()
    assert thumb.exists()
    img = plt.imread(out)
    thumb_img = plt.imread(thumb)
    assert img.shape[0] > thumb_img.shape[0]
    assert img.shape[1] > thumb_img.shape[1]
    ratio = img.shape[1] / img.shape[0]
    ratio_thumb = thumb_img.shape[1] / thumb_img.shape[0]
    assert pytest.approx(ratio, rel=0.05) == 1600 / 1000
    assert pytest.approx(ratio_thumb, rel=0.05) == 1600 / 1000
    plt.close(fig)
