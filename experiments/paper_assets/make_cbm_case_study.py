"""Worked CUB-20 prediction for the paper's concept-bottleneck case study.

The fixed example shows the three properties that distinguish FERL as a
concept-bottleneck head: its named-concept route exposes a detector error, its
bounded-support evidential read-out abstains rather than guessing, and the
route identifies a single concept whose verification restores the correct
classification.

Deterministic (fixed subset/model/detector seed/test instance). Writes
``results/paper_figures/cbm_case_study.pdf`` and a JSON sidecar containing every
number used in the figure and paper text.

Run from the repository root:
    python experiments/paper_assets/make_cbm_case_study.py

If the image path recorded in the manifest is stale, set ``CUB_ROOT`` to the
directory containing CUB's ``images/`` directory.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, "experiments/cub_cbm")
from run_cub_cbm import encoded_splits, read_bundle
from run_cub_cbm_extras import intervention_values, make_model_x
from experiments.cub_cbm.adaptive_intervention import suggest_support_guided_query
from ferl.core.learned_tree import _ds_combine


ART = Path("results/cub_cbm_artifacts/cub_koh112_20")
META = Path("results/cub_cbm_artifacts/20_manifest/test/metadata.jsonl")
FIG = Path("results/paper_figures")
DEPTH, SEED, DET, INST = 12, 0, 0, 284

C_FERL = "#D55E00"
C_FERL_LIGHT = "#F6D7C3"
C_WARN = "#A33A22"
C_WARN_LIGHT = "#F8E5DE"
C_IGN = "#B8B8B8"
C_OTHER = "#E7C6B2"
C_BASE = "#0072B2"
C_TEXT = "#222222"
C_MUTED = "#686868"

DISPLAY_NAMES = {
    "has_bill_shape_hooked_seabird": "hooked-seabird bill",
    "has_bill_shape_all_purpose": "all-purpose bill",
    "has_upperparts_color_grey": "grey upperparts",
    "has_bill_shape_cone": "cone-shaped bill",
    "has_underparts_color_grey": "grey underparts",
    "has_underparts_color_yellow": "yellow underparts",
    "has_upperparts_color_yellow": "yellow upperparts",
}


def pretty(name):
    return DISPLAY_NAMES.get(name, name.removeprefix("has_").replace("_", " "))


def display_bird_name(name):
    return (name
            .replace("Yellow headed", "Yellow-headed")
            .replace("Red winged", "Red-winged")
            .replace("Black footed", "Black-footed")
            .replace("Groove billed", "Groove-billed"))


def load():
    concept_names = json.loads((ART / "concept_names.json").read_text(encoding="utf-8"))
    metadata = [json.loads(line) for line in META.read_text(encoding="utf-8").splitlines()]

    # The artifact labels are the zero-based ``task_labels.species`` values.
    # ``class_id`` is one-based in the CUB metadata and caused the old figure's
    # Red-winged/Rusty/Brewer names to be shifted by one class.
    birds = {
        int(row["task_labels"]["species"]):
        display_bird_name(row["class_name"].split(".", 1)[-1].replace("_", " "))
        for row in metadata
    }
    image_rows = {int(row["image_id"]): row for row in metadata}

    oracle_bundle = read_bundle(ART, source="oracle", subset="20")
    predicted_bundle = read_bundle(ART, source="predicted", detector_seed=DET, subset="20")
    _, oracle_train, _, oracle_test = encoded_splits(oracle_bundle)
    encoder, predicted_train, _, predicted_test = encoded_splits(predicted_bundle)
    encoded_classes = list(encoder.classes_)

    def bird(encoded_label):
        raw_label = int(encoded_classes[int(encoded_label)])
        return birds.get(raw_label, f"class {raw_label}")

    return (
        concept_names,
        bird,
        image_rows,
        (oracle_train, oracle_test),
        (predicted_train, predicted_test),
    )


def resolve_image(row):
    recorded = Path(row["image_path"])
    if recorded.is_file():
        return recorded

    parts = recorded.parts
    try:
        relative = Path(*parts[parts.index("images") + 1:])
    except ValueError as exc:
        raise FileNotFoundError(f"Manifest image path has no images/ component: {recorded}") from exc

    roots = []
    if os.environ.get("CUB_ROOT"):
        roots.append(Path(os.environ["CUB_ROOT"]))
    for root in roots:
        candidate = root / "images" / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {relative}; set CUB_ROOT to the CUB_200_2011 directory"
    )


def masses(ferl, X):
    membership, consequents, names, support = ferl.node_activation_matrix(X)
    keep = np.flatnonzero(ferl.leaf_mask(names))
    _, belief, _, ignorance = _ds_combine(
        membership[:, keep],
        consequents[keep],
        [names[k] for k in keep],
        ferl.C,
        rule="dempster",
        support=support[keep],
    )
    return belief[0], float(ignorance[0])


def dominant_route(ferl, concept_names, oracle_row, predicted_row):
    """Return the maximum-membership route until bounded support vanishes.

    FERL routes fuzzily through both children. For this near-crisp example the
    maximum-membership route has membership one at each displayed split. We
    therefore call it the *dominant* route and stop instead of inventing a hard
    branch when both memberships are zero.
    """
    node, steps = ferl.root_, []
    x = predicted_row[None, :]
    unsupported = None
    while not node["leaf"]:
        feature = int(node["f"])
        low, high = ferl._split(node, x)
        low, high = float(low[0]), float(high[0])
        if max(low, high) <= 1e-12:
            unsupported = {
                "f": feature,
                "name": concept_names[feature],
                "score": float(predicted_row[feature]),
                "support_lo": float(node["lo"]),
                "support_hi": float(node["hi"]),
            }
            break

        take_low = low >= high
        score = float(predicted_row[feature])
        oracle = float(oracle_row[feature])
        steps.append({
            "f": feature,
            "name": concept_names[feature],
            "score": score,
            "oracle": oracle,
            "branch": "low" if take_low else "high",
            "threshold": float(node["center"]),
            "membership": low if take_low else high,
            "misread": (score >= 0.5) != (oracle >= 0.5),
        })
        node = node["L"] if take_low else node["R"]
    return steps, unsupported


def rounded_box(ax, xy, width, height, *, face, edge, linewidth=0.8, radius=0.015):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def draw_figure(image_path, bird, truth, steps, unsupported, flagged,
                bel_raw, ign_raw, bel_fix, ign_fix, cart_pred, lr_pred,
                verified_score):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.15, 3.05), facecolor="white")
    grid = fig.add_gridspec(
        1, 3,
        width_ratios=(1.02, 1.58, 1.50),
        left=0.012, right=0.992, top=0.94, bottom=0.08, wspace=0.16,
    )

    # (a) The actual image makes every later concept statement visually grounded.
    ax_image = fig.add_subplot(grid[0, 0])
    ax_image.imshow(plt.imread(image_path))
    ax_image.set_title("(a) Input image", loc="left", pad=6)
    ax_image.axis("off")
    ax_image.annotate(
        "flagged CUB concept:\nyellow underparts",
        xy=(0.57, 0.47), xycoords="axes fraction",
        xytext=(0.02, 0.93), textcoords="axes fraction",
        ha="left", va="top", fontsize=7.1, color=C_WARN,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_WARN, lw=1.0, alpha=0.94),
        arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.2,
                        connectionstyle="arc3,rad=-0.12"),
    )
    ax_image.text(
        0.5, -0.055, f"True: {truth}", transform=ax_image.transAxes,
        ha="center", va="top", fontsize=7.4, fontweight="bold", color=C_TEXT,
    )

    # (b) Show low/high routing, not the old and incorrect present/absent labels.
    ax_route = fig.add_subplot(grid[0, 1])
    ax_route.set_xlim(0, 1)
    ax_route.set_ylim(0, 1)
    ax_route.axis("off")
    ax_route.set_title("(b) FERL's dominant concept route", loc="left", pad=6)
    ax_route.text(
        0.02, 0.93, "detector score  →  stronger learned branch",
        color=C_MUTED, fontsize=7.0, transform=ax_route.transAxes,
    )
    y0, dy, height = 0.80, 0.105, 0.077
    for rank, step in enumerate(steps):
        y = y0 - rank * dy
        highlighted = step is flagged
        rounded_box(
            ax_route, (0.02, y), 0.96, height,
            face=C_WARN_LIGHT if highlighted else "#F4F4F4",
            edge=C_WARN if highlighted else "#B5B5B5",
            linewidth=1.5 if highlighted else 0.65,
        )
        ax_route.text(
            0.045, y + height / 2, pretty(step["name"]),
            transform=ax_route.transAxes, va="center", fontsize=7.4,
            color=C_TEXT, fontweight="bold" if highlighted else "normal",
        )
        suffix = "  ×" if highlighted else ""
        ax_route.text(
            0.955, y + height / 2,
            f'{step["score"]:.2f}  →  {step["branch"]}{suffix}',
            transform=ax_route.transAxes, va="center", ha="right", fontsize=7.2,
            color=C_WARN if highlighted else C_MUTED,
            fontweight="bold" if highlighted else "normal",
        )
        if rank < len(steps) - 1:
            ax_route.annotate(
                "", xy=(0.50, y - 0.020), xytext=(0.50, y - 0.002),
                xycoords=ax_route.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#A0A0A0", lw=0.7),
            )

    flagged_y = y0 - steps.index(flagged) * dy
    ax_route.text(
        0.045, flagged_y - 0.025,
        f'annotation: present; split at {flagged["threshold"]:.2f}',
        transform=ax_route.transAxes, fontsize=6.7, color=C_WARN, style="italic",
    )
    ax_route.annotate(
        "", xy=(0.50, 0.153), xytext=(0.50, flagged_y - 0.034),
        xycoords=ax_route.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.0),
    )
    rounded_box(
        ax_route, (0.08, 0.035), 0.84, 0.105,
        face="#EFEFEF", edge="#8C8C8C", linewidth=0.9,
    )
    ax_route.text(
        0.50, 0.105, "low branch has no supported leaf",
        transform=ax_route.transAxes, ha="center", va="center",
        fontsize=7.3, fontweight="bold", color=C_TEXT,
    )
    ax_route.text(
        0.50, 0.067, "all compatible leaf firings = 0",
        transform=ax_route.transAxes, ha="center", va="center",
        fontsize=6.7, color=C_MUTED,
    )

    # (c) Contrast the raw prediction with the result of one named intervention.
    ax_out = fig.add_subplot(grid[0, 2])
    ax_out.set_xlim(0, 1)
    ax_out.set_ylim(0, 1)
    ax_out.axis("off")
    ax_out.set_title("(c) Abstain, verify, decide", loc="left", pad=6)

    ax_out.text(0.02, 0.89, "Raw detector concepts", transform=ax_out.transAxes,
                fontsize=7.8, fontweight="bold", color=C_TEXT)
    ax_out.barh(0.80, 0.94, left=0.02, height=0.105, color=C_IGN, edgecolor="white")
    ax_out.text(0.49, 0.80, f"ignorance {ign_raw:.2f}", transform=ax_out.transAxes,
                ha="center", va="center", fontsize=7.5, color=C_TEXT)
    ax_out.text(0.02, 0.715, "FERL → abstain", transform=ax_out.transAxes,
                fontsize=7.5, fontweight="bold", color=C_FERL)
    ax_out.text(0.02, 0.655, f"CART → {cart_pred}  ×", transform=ax_out.transAxes,
                fontsize=7.0, color=C_BASE)
    ax_out.text(0.02, 0.602, f"linear → {lr_pred}  ×", transform=ax_out.transAxes,
                fontsize=7.0, color=C_BASE)

    ax_out.annotate(
        "backtrack to nearest supported branch", xy=(0.49, 0.445), xytext=(0.49, 0.535),
        xycoords=ax_out.transAxes, textcoords=ax_out.transAxes,
        ha="center", va="center", fontsize=7.0, color=C_WARN,
        arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.0),
    )
    rounded_box(
        ax_out, (0.05, 0.365), 0.89, 0.075,
        face=C_WARN_LIGHT, edge=C_WARN, linewidth=1.1,
    )
    ax_out.text(
        0.495, 0.403,
        f'yellow underparts: {flagged["score"]:.2f} → {verified_score:.2f}',
        transform=ax_out.transAxes, ha="center", va="center",
        fontsize=7.0, color=C_WARN, fontweight="bold",
    )

    ax_out.text(0.02, 0.285, "After one verification", transform=ax_out.transAxes,
                fontsize=7.8, fontweight="bold", color=C_TEXT)
    top = int(np.argmax(bel_fix))
    true_width = 0.94 * float(bel_fix[top])
    ax_out.barh(0.195, true_width, left=0.02, height=0.105,
                color=C_FERL, edgecolor="white")
    ax_out.barh(0.195, 0.94 - true_width, left=0.02 + true_width, height=0.105,
                color=C_OTHER, edgecolor="white")
    ax_out.text(
        0.02 + true_width / 2, 0.195,
        f"Yellow-headed\nbelief {bel_fix[top]:.2f}",
        transform=ax_out.transAxes, ha="center", va="center",
        fontsize=6.7, color="white", fontweight="bold",
    )
    ax_out.text(
        0.02 + true_width + (0.94 - true_width) / 2, 0.195,
        "other\nspecies", transform=ax_out.transAxes,
        ha="center", va="center", fontsize=6.4, color=C_TEXT,
    )
    ax_out.text(
        0.02, 0.105,
        f"FERL → correct; ignorance {ign_fix:.2f}",
        transform=ax_out.transAxes, fontsize=7.1, color=C_FERL, fontweight="bold",
    )

    fig.savefig(FIG / "cbm_case_study.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    concept_names, bird, image_rows, (otr, ote), (ptr, pte) = load()
    ferl = make_model_x("ferl-deep", SEED, DEPTH).fit(ptr.C, ptr.y)
    cart = DecisionTreeClassifier(
        random_state=SEED, min_samples_split=5, min_samples_leaf=2,
    ).fit(ptr.C, ptr.y)
    linear = LogisticRegression(max_iter=2000, random_state=SEED).fit(ptr.C, ptr.y)
    value_absent, value_present = intervention_values(otr.C, ptr.C)

    instance, truth_index = INST, int(pte.y[INST])
    steps, unsupported = dominant_route(
        ferl, concept_names, ote.C[instance], pte.C[instance],
    )
    belief_raw, ignorance_raw = masses(ferl, pte.C[instance:instance + 1])

    # Choose what to ask using only FERL's unsupported route. The annotation is
    # revealed below only after the policy has selected a concept, simulating a
    # human response rather than leaking oracle information into query choice.
    suggestion = suggest_support_guided_query(
        ferl,
        pte.C[instance],
        queried=np.zeros(pte.C.shape[1], dtype=bool),
        value_absent=value_absent,
        value_present=value_present,
    )
    if suggestion is None:
        raise RuntimeError("support-guided policy found no concept to query")
    feature = suggestion.feature
    flagged = next(step for step in steps if step["f"] == feature)

    fixed = pte.C[instance:instance + 1].copy()
    fixed[0, feature] = (
        value_present[feature] if ote.C[instance, feature] >= 0.5 else value_absent[feature]
    )
    belief_fixed, ignorance_fixed = masses(ferl, fixed)
    cart_index = int(cart.predict(pte.C[instance:instance + 1])[0])
    linear_index = int(linear.predict(pte.C[instance:instance + 1])[0])

    image_id = int(pte.ids[instance])
    image_path = resolve_image(image_rows[image_id])
    truth = bird(truth_index)
    cart_pred, linear_pred = bird(cart_index), bird(linear_index)

    # Guard the scientific claim against silently changing artifacts or models.
    assert truth == "Yellow-headed Blackbird"
    assert pretty(flagged["name"]) == "yellow underparts"
    assert suggestion.reason == "support_backtrack"
    assert unsupported is not None
    assert ignorance_raw > 0.999 and ignorance_fixed < 1e-9
    assert int(np.argmax(belief_fixed)) == truth_index
    assert cart_index != truth_index and linear_index != truth_index

    # Certify the caption's instance-level minimality claim. With no edit the
    # output is vacuous; among all 112 possible single verified-concept edits,
    # this is the unique one that yields the correct supported prediction.
    single_repair_features = []
    verified_row = np.where(ote.C[instance] >= 0.5, value_present, value_absent)
    for candidate in range(pte.C.shape[1]):
        candidate_x = pte.C[instance:instance + 1].copy()
        candidate_x[0, candidate] = verified_row[candidate]
        candidate_belief, candidate_ignorance = masses(ferl, candidate_x)
        if (candidate_ignorance < 1e-9
                and int(np.argmax(candidate_belief)) == truth_index):
            single_repair_features.append(candidate)
    assert single_repair_features == [feature]

    draw_figure(
        image_path=image_path,
        bird=bird,
        truth=truth,
        steps=steps,
        unsupported=unsupported,
        flagged=flagged,
        bel_raw=belief_raw,
        ign_raw=ignorance_raw,
        bel_fix=belief_fixed,
        ign_fix=ignorance_fixed,
        cart_pred=cart_pred,
        lr_pred=linear_pred,
        verified_score=float(fixed[0, feature]),
    )

    numbers = {
        "instance": instance,
        "image_id": image_id,
        "image_path": str(image_path),
        "truth": truth,
        "raw_ignorance": round(ignorance_raw, 3),
        "verified_ignorance": round(ignorance_fixed, 3),
        "verified_belief": round(float(belief_fixed.max()), 3),
        "cart": cart_pred,
        "linear": linear_pred,
        "flagged_concept": pretty(flagged["name"]),
        "query_policy": suggestion.reason,
        "unique_correct_single_concept_repair": True,
        "flagged_score": round(flagged["score"], 3),
        "verified_score": round(float(fixed[0, feature]), 3),
        "dominant_route_length": len(steps),
        "unsupported_at": pretty(unsupported["name"]),
    }
    (FIG / "cbm_case_study.txt").write_text(
        json.dumps(numbers, indent=1) + "\n", encoding="utf-8",
    )
    print("wrote cbm_case_study.pdf")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
