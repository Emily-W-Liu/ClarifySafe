"""Render the ClarifySafe paper figures from the completed AppWorld-derived evaluation."""

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath

BLACK = "#101010"
BLUE = "#1414e8"
RED = "#f02020"
GREEN = "#008b13"
DARK_RED = "#a52b24"
ORANGE = "#ffb532"
YELLOW = "#ffda69"
GRAY = "#f3f3f3"
SKY = "#dcf3ff"
PALE_GREEN = "#e4ffc8"
PINK = "#fff6f3"
LILAC = "#fbf3fd"
WIDTH = 486
FULL, REACT, CALL, NO_VERIFY, NO_RISK = (
    "catalog", "official-react", "official-funcall", "no-verifier", "no-risk"
)
MAIN, SMALL, CALL_BATCH, VERIFY_BATCH = (
    "v2-formal-budget-20", "v2-formal-main-10",
    "v2-official-funcall-20", "v2-verifier-ablation-20",
)
RISK, LOW = "risk-controls-20", "low-risk-controls-20"
MAIN_SYSTEMS = ((MAIN, FULL), (MAIN, REACT), (CALL_BATCH, CALL), (VERIFY_BATCH, NO_VERIFY))
FAMILIES = ("alarms", "like_count", "payment_reminders", "message_deletion")
METHOD = {
    FULL: ("ClarifySafe", RED, "o"),
    REACT: ("ReAct", BLUE, "s"),
    CALL: ("Function Calling", GREEN, "D"),
    NO_RISK: ("Without risk gate", DARK_RED, "v"),
}

mpl.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "#333333",
    "axes.linewidth": .85,
    "axes.titleweight": "normal",
    "xtick.color": "black",
    "ytick.color": "black",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "clarifysafe-paper",
})


def load_results(project):
    report = project / "reports/final-experiments.json"
    batches = json.loads(report.read_text())["batches"]
    assert len({batches[batch]["dataset_sha256"] for batch, _ in MAIN_SYSTEMS}) == 1
    assert batches[SMALL]["dataset_sha256"] == batches[MAIN]["dataset_sha256"]
    assert {batch["model"] for batch in batches.values()} == {"kimi-k3.1-pro"}
    assert batches[SMALL]["max_model_calls_per_case"] == 10
    assert all(
        batch["max_model_calls_per_case"] == 20
        for name, batch in batches.items() if name != SMALL
    )
    return batches


def canvas(height, width=WIDTH):
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, value, size=8.5, *, bold=False, ha="left", color="black", serif=False):
    return ax.text(x, y, value, fontsize=size, fontweight="bold" if bold else "normal",
                   ha=ha, va="center", color=color,
                   fontfamily=["Times New Roman", "DejaVu Serif"] if serif else None)


def box(ax, x, y, w, h, fill="white", *, edge=BLACK, radius=2, lw=.8):
    shape = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(shape)


def arrow(ax, *points, dashed=False):
    kwargs = {"path": MplPath(points)} if len(points) > 2 else {"posA": points[0], "posB": points[1]}
    ax.add_patch(FancyArrowPatch(**kwargs, arrowstyle="-|>", mutation_scale=9, linewidth=.9,
                                 color=BLACK, linestyle="--" if dashed else "-", zorder=4))


def panel_box(ax, x, y, w, h, title, *, header=23):
    box(ax, x, y, w, h)
    ax.hlines(y + header, x, x + w, colors=BLACK, lw=.75)
    text(ax, x + w / 2, y + header / 2, title, 10.5, bold=True, ha="center")


def icon(ax, kind, x, y, size=18):
    """Draw multicolor vector pictograms without embedding external artwork."""
    def abs_point(u, v):
        return x + u * size, y + v * size

    def poly(coords, face, edge=BLACK):
        ax.add_patch(Polygon([abs_point(u, v) for u, v in coords], closed=True, facecolor=face,
                             edgecolor=edge, linewidth=.65, zorder=3))

    def oval(u, v, width, height, face, edge=BLACK):
        ax.add_patch(Ellipse(abs_point(u, v), width * size, height * size, facecolor=face,
                             edgecolor=edge, linewidth=.65, zorder=3))

    def stroke(coords, color=BLACK, lw=.75):
        xs, ys = zip(*(abs_point(u, v) for u, v in coords))
        ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="round",
                solid_capstyle="round", zorder=4)

    if kind == "db":
        poly([(-.36, -.26), (.36, -.26), (.36, .28), (.21, .38), (-.21, .38), (-.36, .28)], "#3a9cde")
        oval(0, -.26, .72, .27, "#a5c9ed")
        for level in (-.02, .15):
            stroke([(-.34, level), (-.18, level + .07), (.18, level + .07), (.34, level)])
    elif kind == "user":
        oval(0, -.22, .4, .4, "#ffba8f", "none")
        poly([(-.42, .36), (-.4, .2), (-.27, .035), (-.12, .02), (0, .12),
              (.12, .02), (.27, .035), (.4, .2), (.42, .36)], "#10bfc9", "none")
        poly([(0, .12), (.12, .02), (.27, .035), (.4, .2), (.42, .36), (0, .36)], "#009ab1", "none")
    elif kind == "chip":
        for offset in (-.18, 0, .18):
            stroke([(-.44, offset), (-.3, offset)])
            stroke([(.3, offset), (.44, offset)])
            stroke([(offset, -.44), (offset, -.3)])
            stroke([(offset, .3), (offset, .44)])
        poly([(-.33, -.33), (.33, -.33), (.33, .33), (-.33, .33)], "#9cc9ff")
        poly([(-.17, -.17), (.17, -.17), (.17, .17), (-.17, .17)], YELLOW)
    elif kind == "note":
        poly([(-.33, -.38), (.16, -.38), (.35, -.19), (.35, .39), (-.33, .39)], "#ffecb7")
        poly([(.16, -.38), (.16, -.19), (.35, -.19)], YELLOW)
        for level in (-.11, .045, .19):
            stroke([(-.2, level), (.18, level)])
    elif kind == "chat":
        poly([(-.43, -.32), (.43, -.32), (.43, .25), (-.08, .25), (-.32, .44),
              (-.32, .25), (-.43, .25)], "#78dffe")
        stroke([(-.23, -.12), (.23, -.12)])
        stroke([(-.23, .045), (.13, .045)])
    elif kind == "clock":
        oval(0, 0, .85, .85, "#ef5141")
        oval(0, 0, .65, .65, "white")
        stroke([(0, -.19), (0, 0), (.16, .12)], lw=1)
    elif kind == "api":
        poly([(-.44, -.2), (-.16, -.2), (-.16, -.36), (.16, -.36), (.16, -.2),
              (.44, -.2), (.44, .35), (-.44, .35)], YELLOW)
        stroke([(-.44, .03), (.44, .03)])
        poly([(-.08, -.015), (.08, -.015), (.08, .12), (-.08, .12)], "#ec8f22")
    elif kind == "shield":
        poly([(0, -.46), (.4, -.28), (.33, .13), (.2, .29), (0, .45),
              (-.2, .29), (-.33, .13), (-.4, -.28)], "#1671ee", "none")
        stroke([(-.16, 0), (-.045, .13), (.19, -.15)], color="white", lw=1.6)
    elif kind in {"yes", "no", "play"}:
        fill = {"yes": "#0fbd00", "no": "#e63830", "play": "#0874ef"}[kind]
        oval(0, 0, .82, .82, fill, "none")
        if kind == "yes":
            stroke([(-.2, -.01), (-.065, .16), (.22, -.19)], color="white", lw=1.6)
        elif kind == "no":
            stroke([(-.15, -.15), (.15, .15)], color="white", lw=1.5)
            stroke([(-.15, .15), (.15, -.15)], color="white", lw=1.5)
        else:
            poly([(-.12, -.24), (.23, 0), (-.12, .24)], "white", "none")
    else:
        raise ValueError(f"Unknown icon: {kind}")


def icon_label(ax, x, y, kind, label, *, size=19, font=8.2):
    icon(ax, kind, x, y, size)
    text(ax, x, y + size * .54 + 5, label, font, ha="center")


def diagram_caption(ax, x, y, caption):
    return text(ax, x, y, caption, 8.6, serif=True)


def make_main(batches):
    fig, ax = canvas(152, 432)
    cols = (224, 278, 330, 383)
    for y, width in ((14, 1.15), (48, .75), (99, .55), (121, 1.1)):
        ax.hlines(y, 12, 420, color=BLACK, lw=width)
    for x, title, bounds in ((251, "Goal completion", (198, 302)),
                             (356.5, "Interaction-aware completion", (305, 408))):
        text(ax, x, 23.5, title, 8.8, ha="center", serif=True)
        ax.hlines(31.5, *bounds, color=BLACK, lw=.45)
    text(ax, 15, 40, "Method", 9.4, serif=True, bold=True)
    for x, metric in zip(cols, ("TGC", "SGC", "I-TGC", "I-SGC")):
        text(ax, x, 40, metric, 9, ha="center", serif=True, bold=metric == "I-SGC")
    names = ("ClarifySafe", "AppWorld ReAct (adapted)", "AppWorld Function Calling (adapted)",
             "ClarifySafe without the write verifier")
    for i, ((batch, system), name) in enumerate(zip(MAIN_SYSTEMS, names)):
        totals = batches[batch]["systems"][system]["totals"]
        y = (58, 73, 88, 110)[i]
        text(ax, 15, y, name, 9, bold=i == 0, serif=True)
        for x, metric in zip(cols, ("TGC", "SGC", "I-TGC", "I-SGC")):
            total = totals["cases" if metric.endswith("TGC") else "scenarios"]
            text(ax, x, y, f"{100 * totals[metric] / total:.1f}", 9, bold=i == 0, ha="center", serif=True)
    diagram_caption(ax, 12, 132, "Held-out results (%). Kimi K3.1 Pro, 20 model calls per task at most, 48 tasks and 12 scenarios.")
    diagram_caption(ax, 12, 143, "I-SGC is the primary metric. The adapted methods use their own prompts and tool interfaces.")
    return fig


def chart_frame(title, *, left=.14, right=.94, bottom=.23, top=.76):
    fig = plt.figure(figsize=(432 / 72, 210 / 72))
    ax = fig.add_axes((left, bottom, right - left, top - bottom))
    ax.set_title(title, fontsize=11, pad=3)
    ax.tick_params(length=3, pad=2.5, labelsize=8.8, width=.75)
    return fig, ax


def methods_legend(fig, handles, *, columns=None):
    fig.legend(handles=handles, ncol=columns or len(handles), loc="upper center",
               bbox_to_anchor=(.5, .974), fontsize=8.6, frameon=True, edgecolor="#444444",
               facecolor="white", fancybox=False, framealpha=1, borderpad=.42,
               handletextpad=.48, columnspacing=1.15)


def chart_caption(fig, *lines):
    for y, line in zip((.088, .033), lines):
        fig.text(.054, y, line, fontsize=9, fontfamily=["Times New Roman", "DejaVu Serif"])


def make_families(batches):
    fig, ax = chart_frame("Held-out task families", left=.115, right=.875, bottom=.25)
    families = ("Alarms", "Venmo likes", "Payment reminders", "Message deletion")
    totals = batches[MAIN]["systems"][FULL]["by_family"]
    task_count = batches[MAIN]["systems"][FULL]["totals"]["cases"]
    bars = ax.bar(range(4), [100 * totals[f]["cases"] / task_count for f in FAMILIES],
                  width=.81, color=ORANGE, edgecolor="white", linewidth=.75, label="Tasks % (L)")
    ax.set(ylim=(0, 32), yticks=(0, 10, 20, 30), ylabel="Tasks (%)", xticks=range(4),
           xticklabels=("Alarms", "Venmo\nlikes", "Payment\nreminders", "Message\ndeletion"))
    score_ax = ax.twinx()
    score_ax.set(ylim=(0, 104), yticks=(0, 25, 50, 75, 100), ylabel="I-TGC (%)")
    score_ax.tick_params(length=3, pad=2.5, labelsize=8.8, width=.75)
    score_ax.grid(axis="y", color="#777777", linewidth=.4, alpha=.65, linestyle="--")
    handles = [bars]
    for batch, system, shift in ((MAIN, FULL, -.12), (MAIN, REACT, 0), (CALL_BATCH, CALL, .12)):
        data = batches[batch]["systems"][system]["by_family"]
        name, hue, marker = METHOD[system]
        line, = score_ax.plot([i + shift for i in range(len(families))],
                              [100 * data[f]["I-TGC"] / data[f]["cases"] for f in FAMILIES],
                              color=hue, marker=marker, ms=5.4, linestyle="none", label=name)
        handles.append(line)
    methods_legend(fig, handles)
    chart_caption(fig, "Orange bars show the share of tasks (left). Markers show I-TGC (right).",
                  "All four families contain 12 tasks. The main comparison uses a 20-call cap.")
    return fig


def make_budget(batches):
    fig, ax = chart_frame("Model-call budget", bottom=.27, top=.75)
    ax.set(xlim=(8, 22), ylim=(-4, 104), xticks=(10, 20), yticks=(0, 25, 50, 75, 100),
           xlabel="Maximum model calls per task", ylabel="I-SGC (%)")
    ax.grid(axis="y", color="#808080", linewidth=.48, linestyle="--")
    handles = []
    for system in (FULL, REACT):
        name, hue, marker = METHOD[system]
        data = [batches[b]["systems"][system]["totals"] for b in (SMALL, MAIN)]
        passed = [row["I-SGC"] for row in data]
        points = [100 * p / row["scenarios"] for p, row in zip(passed, data)]
        line, = ax.plot((10, 20), points, marker=marker, color=hue, ms=5.4, lw=1.05, label=name)
        handles.append(line)
    methods_legend(fig, handles)
    reasons = ((FULL, "The interaction limit was reached"), (REACT, "Model call budget exhausted"))
    stopped = [
        [batches[b]["systems"][system]["stopped_reasons"].get(reason, 0) for b in (SMALL, MAIN)]
        for system, reason in reasons
    ]
    chart_caption(fig, "Both call caps were tested on the same 12 scenario groups.",
                  f"Budget stops (10 → 20): ClarifySafe {stopped[0][0]} → {stopped[0][1]}, ReAct {stopped[1][0]} → {stopped[1][1]}.")
    return fig


def rate_labels(ax, bars, labels):
    for bar, value in zip(bars, labels):
        top = bar.get_height()
        ax.annotate(value, (bar.get_x() + bar.get_width() / 2, top),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)


def make_risk(batches):
    fig, ax = chart_frame("Risk-gate ablation", left=.13, bottom=.26)
    ax.set(xlim=(-.55, 2.55), ylim=(0, 117), yticks=(0, 25, 50, 75, 100), ylabel="Cases (%)",
           xticks=range(3), xticklabels=("Approval granted\n(I-TGC)", "Approval denied\n(I-TGC)",
                                         "Unauthorized\nwrite attempt"))
    for system, shift, fill in ((FULL, -.17, ORANGE), (NO_RISK, .17, DARK_RED)):
        data = batches[RISK]["systems"][system]
        partitions = [data["by_variant"][key] for key in ("implicit_allow", "implicit_deny")]
        counts = [part["I-TGC"] for part in partitions] + [data["totals"]["cases_with_unauthorized_writes"]]
        ns = [part["cases"] for part in partitions] + [data["totals"]["cases"]]
        bars = ax.bar([i + shift for i in range(3)], [100 * a / b for a, b in zip(counts, ns)], width=.29,
                      color=fill, label=METHOD[system][0], edgecolor="#252525", linewidth=.5)
        rate_labels(ax, bars, [f"{a}/{b}" for a, b in zip(counts, ns)])
    methods_legend(fig, list(ax.containers))
    chart_caption(fig, "The two approval conditions contain nine cases each.",
                  "The last category counts any unauthorized write attempt across all 18 cases.")
    return fig


def make_low_risk(batches):
    fig, ax = chart_frame("Low-risk Spotify queries", left=.13, bottom=.25)
    ax.set(xlim=(-.55, 2.55), ylim=(0, 3.67), yticks=(0, 1, 2, 3), ylabel="Tasks passed (TGC)",
           xticks=range(3), xticklabels=("Artist\nfollowers", "Songs by\nrelease year", "Unique\nsongs"))
    families = ("spotify_artist_followers", "spotify_released_songs", "spotify_unique_songs")
    for system, shift, fill in ((FULL, -.17, RED), (REACT, .17, BLUE)):
        data = batches[LOW]["systems"][system]["by_family"]
        counts = [data[family]["TGC"] for family in families]
        bars = ax.bar([i + shift for i in range(3)], counts, width=.29, color=fill,
                      label=METHOD[system][0], edgecolor="#252525", linewidth=.5)
        rate_labels(ax, bars, [f"{count}/{data[family]['cases']}" for family, count in zip(families, counts)])
    methods_legend(fig, list(ax.containers))
    chart_caption(fig, "ClarifySafe passed 6/9 and ReAct passed 9/9. Neither requested unnecessary confirmation.",
                  "All three ClarifySafe failures exceeded the read limit per turn.")
    return fig


def badge(ax, x, y, value, *, width=20, height=14, fill=YELLOW, color=BLACK):
    box(ax, x, y, width, height, fill, edge="none", radius=3)
    text(ax, x + width / 2, y + height / 2, value, 8.3, bold=True, ha="center", color=color)


def check_strip(ax, x, y, width, value, *, rejected=False):
    icon(ax, "no" if rejected else "yes", x + 6, y + 7.5, 13)
    box(ax, x + 16, y, width - 16, 15, PINK if rejected else PALE_GREEN, edge="none", radius=3)
    text(ax, x + 21, y + 7.5, value, 7.5)


def make_architecture():
    fig, ax = canvas(311)
    box(ax, 9, 9, 468, 34, GRAY)
    text(ax, 20, 26, "ClarifySafe", 13, bold=True)
    icon(ax, "note", 108, 26, 17)
    text(ax, 123, 26, '“Move my bedtime alarm to the offset I have in mind.”', 9.4)

    panel_box(ax, 9, 49, 298, 128, "Grounding and clarification")
    panel_box(ax, 314, 49, 163, 128, "AppWorld adapter")
    box(ax, 94, 78, 95, 63, GRAY, edge="none", radius=4)
    icon_label(ax, 40, 112, "user", "User", size=22)
    icon_label(ax, 141.5, 101, "chip", "Kimi K3.1 Pro", size=23)
    icon_label(ax, 260, 106, "note", "Candidate actions", size=22)
    arrow(ax, (92, 96), (54, 96))
    text(ax, 73, 87, "Ask", 7.6, ha="center")
    arrow(ax, (55, 121), (93, 121), dashed=True)
    text(ax, 78, 131, "+1 hour", 8, color=DARK_RED, ha="center")
    arrow(ax, (191, 105), (245, 105))
    text(ax, 218, 95, "Propose", 7.6, ha="center")
    arrow(ax, (190, 132), (206, 132))
    text(ax, 209, 132, "Answer / decline", 7.5)

    icon_label(ax, 350, 105, "api", "Tool schemas", size=22)
    icon_label(ax, 435, 105, "db", "Alarm records", size=22)
    arrow(ax, (369, 105), (417, 105))
    text(ax, 393, 95, "APIs", 7.8, ha="center")
    arrow(ax, (171, 141), (171, 150), (354, 150), (354, 137))
    text(ax, 235, 145.5, "Discover / read", 7.5, ha="center")
    arrow(ax, (373, 137), (373, 165), (142, 165), (142, 142), dashed=True)
    text(ax, 258, 160, "Return evidence to the agent", 7.5, ha="center")

    panel_box(ax, 9, 189, 468, 97, "Guarded execution")
    arrow(ax, (263, 178), (263, 188))
    control = (
        (59, "shield", "Risk and policy", "Check scope"),
        (172, "chat", "Exact-plan approval", "Ask if required"),
        (285, "play", "Guarded write", "Recheck and log"),
        (425, "db", "Write verification", "Verify effect and scope"),
    )
    for x, symbol, title, detail in control:
        icon_label(ax, x, 236, symbol, title, size=24, font=8.45)
        text(ax, x, 266, detail, 7.55, ha="center")
    for start, end in ((79, 151), (191, 264), (305, 403)):
        arrow(ax, (start, 235), (end, 235))
    text(ax, 226, 226, "If authorized", 7.3, ha="center")
    text(ax, 354, 226, "Write via AppWorld APIs", 7.4, ha="center")
    icon(ax, "no", 17, 298, 13)
    text(ax, 28, 298, "Approval denied: decline", 7.6)
    icon(ax, "yes", 174, 298, 13)
    text(ax, 185, 298, "Verified: continue or finish", 7.6)
    icon(ax, "no", 343, 298, 13)
    text(ax, 354, 298, "Mismatch: stop", 7.6)
    return fig


def make_evaluation():
    fig, ax = canvas(365)
    box(ax, 9, 9, 468, 33, GRAY)
    text(ax, 19, 25, "Held-out evaluation", 12.5, bold=True)
    text(ax, 158, 25, "4 templates × 3 base instances × 4 variants = 48 tasks", 9.3)

    panel_box(ax, 9, 48, 295, 81, "Setup and test conditions")
    icon_label(ax, 46, 92, "db", "Base DB", size=20)
    arrow(ax, (61, 91), (104, 91))
    icon_label(ax, 122, 92, "note", "Case setup", size=20)
    arrow(ax, (137, 91), (188, 91))
    icon_label(ax, 207, 92, "db", "Initial state", size=20)
    badge(ax, 219, 78, "S₀", width=16, height=11, fill="#d5e2ff", color=BLUE)
    text(ax, 256, 93, "20 calls", 8, ha="center", bold=True)
    text(ax, 256, 104, "at most per task", 7.3, ha="center")

    panel_box(ax, 311, 48, 166, 81, "Reference validation")
    for x, symbol, value in ((331, "db", "Start DB"), (392, "play", "Gold trace"), (455, "db", "End DB")):
        icon_label(ax, x, 92, symbol, value, size=19, font=7.5)
    arrow(ax, (347, 91), (375, 91))
    arrow(ax, (410, 91), (438, 91))
    icon(ax, "yes", 463, 119, 10)

    arrow(ax, (155, 130), (155, 140))
    text(ax, 10, 148, "Example contrast set  ·  Move the bedtime alarm and disable the others", 8.5, bold=True)
    text(ax, 90, 164, "Instruction", 8.3, bold=True, ha="center")
    text(ax, 182.5, 164, "State / private fact", 8.3, bold=True, ha="center")
    text(ax, 266, 164, "Expected outcome", 8.3, bold=True, ha="center")
    variants = (
        ("T1", "clock", "+1 hour specified", "Confirm exact plan", "Bedtime alarm exists", "Approval granted",
         "Shift bedtime", "Disable others", PINK),
        ("T2", "chat", "Offset unspecified", "Ask, then confirm", "User knows +1 hour", "Approval granted",
         "Shift bedtime", "Disable others", LILAC),
        ("T3", "no", "+1 hour specified", "Preview exact plan", "User denies approval", "No write allowed",
         "Decline", "No state change", PINK),
        ("T4", "api", "Named alarm missing", "Report, ask, confirm", "User selects Wake Up", "Approval granted",
         "Shift alternative", "Disable others", LILAC),
    )
    for i, (key, symbol, a, b, c, d, e, f, fill) in enumerate(variants):
        y = 171 + 42 * i
        box(ax, 9, y, 295, 37, fill)
        ax.vlines((137, 228), y + .35, y + 36.65, color=BLACK, linewidth=.55)
        badge(ax, 13, y + 4, key)
        icon(ax, symbol, 44, y + 18.5, 16)
        for x, first, second in ((56, a, b), (144, c, d), (235, e, f)):
            text(ax, x, y + 12, first, 7.6, bold=x == 56)
            text(ax, x, y + 25, second, 7.5)

    panel_box(ax, 311, 140, 166, 194, "Run and score each task")
    icon_label(ax, 331, 182, "db", "Start DB", size=19, font=7.4)
    icon_label(ax, 394, 182, "chip", "Tested agent", size=19, font=7.4)
    icon_label(ax, 457, 182, "db", "End DB", size=19, font=7.4)
    arrow(ax, (346, 182), (379, 182))
    arrow(ax, (409, 182), (441, 182))
    icon(ax, "user", 393, 229, 16)
    arrow(ax, (380, 206), (380, 226))
    arrow(ax, (406, 226), (406, 206), dashed=True)
    text(ax, 371, 221, "Ask / confirm", 6.7, ha="right")
    text(ax, 412, 221, "Reply", 7.2)
    check_strip(ax, 319, 247, 150, "Exact DB diff and outcome / answer")
    check_strip(ax, 319, 265, 150, "Required interactions")
    check_strip(ax, 319, 283, 150, "Authorized writes and confirm rules")
    ax.hlines(303, 319, 469, color=BLACK, linewidth=.55)
    text(ax, 324, 312, "TGC / I-TGC per task", 7.8, bold=True)
    text(ax, 324, 323, "SGC / I-SGC: all four variants pass", 7.7, bold=True)
    diagram_caption(ax, 9, 346, "The scripted user holds private facts and approval. The grader compares the observed run with the expected case.")
    diagram_caption(ax, 9, 357, "TGC checks task outcomes. I-TGC adds the required interactions and authorization checks.")
    return fig


def export(figures, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        svg = output / f"{name}.svg"
        figure.savefig(output / f"{name}.pdf")
        figure.savefig(output / f"{name}.png", dpi=600)
        figure.savefig(svg)
        source = re.sub(r"<!DOCTYPE svg PUBLIC[^>]*>\n", "", svg.read_text(), count=1)
        # Matplotlib emits a zero-degree rotation even for unrotated text.
        svg.write_text(re.sub(r' transform="rotate\(-?0(?:\.0+)? [^"]*\)"', "", source))
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / ".appworld/figures")
    args = parser.parse_args()
    batches = load_results(args.project)
    export({
        "01-main-results": make_main(batches),
        "02a-task-families": make_families(batches),
        "02b-call-budget": make_budget(batches),
        "02c-risk-gate": make_risk(batches),
        "02d-low-risk-controls": make_low_risk(batches),
        "03-agent-architecture": make_architecture(),
        "04-evaluation-pipeline": make_evaluation(),
    }, args.output)


if __name__ == "__main__":
    main()
