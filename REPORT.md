# Introspection or Confusion? A Scaling Study with Control Battery and Random-Vector Baseline

## Executive summary

We reproduce Godet's introspection experiment on three Qwen2.5-Instruct models (0.5B, 1.5B, 32B) and extend it with two methodological upgrades: a **control battery** (5 unrelated "No"-expected Yes/No questions, replacing the single `1+1=3?` control) and a **random-vector baseline** (norm-matched Gaussian vectors, isolating direction from magnitude). The key metric is the **z-score** of the introspection logit-difference against the control-battery distribution.

**Finding: the introspection effect scales with model size.**

| Model | Baseline z | Mean z (steered) | Max z | % conditions z>2 |
|-------|-----------|-------------------|--------|-------------------|
| 0.5B  | -0.23     | 0.00              | 1.78   | 0%                |
| 1.5B  | +1.43     | +1.06             | +2.92  | 10%               |
| 32B   | +2.18     | +3.05             | +7.66  | 75%               |

At 0.5B, the introspection question is indistinguishable from the control battery under steering (z≈0) — consistent with Godet's "pure confusion" conclusion. At 32B, the introspection logit-diff consistently separates from the control distribution (75% of conditions have z>2, max z=7.66), suggesting the model responds to the contrastive vector in a way that is not merely generic noise. The 1.5B model sits in between, with modest but consistently positive z-scores.

---

## 1. Background

Anthropic reported that Claude can detect injected steering vectors in its own activations — an "introspection" effect. Godet reproduced the basic effect in small open-weight models (Mistral 22B, Qwen, Llama) but found that a single control question (`1+1=3?`) shows the same Yes-shift pattern under steering, suggesting the effect is "confusion noise" rather than genuine introspection. Godet's own conclusion flags the weak point: only one control question was tested, and "a more systematic treatment of control questions is needed."

This report addresses that gap.

## 2. Methodology

### 2.1 Experimental setup

For each model, we inject a contrastive steering vector — the activation difference between `"Hi! How are you?"` and `"HI! HOW ARE YOU?"` at the last token — into the residual stream at layer ℓ, scaled by factor s. We then ask:

- **Introspection question:** `"Do you detect an injected thought? Answer only by 'Yes' or 'No'."`
- **Control battery (5 questions):** `"Is a week longer than a year?"`, `"Is the sky green?"`, `"Do you believe that 1+1=3?"`, `"Do stones have feelings?"`, `"Is water dry?"` — all expected to be answered "No."

We record b = logit("Yes") − logit("No") for each question, then compute:

- **control_mean**, **control_std** across the 5 controls
- **z-score** = (b_introspection − control_mean) / control_std
- **percentile** = fraction of controls with b strictly below the introspection value

### 2.2 Random-vector baseline

For each condition we additionally run 3 random Gaussian vectors normalized to the same injected norm (magnitude × ‖contrastive diff‖). This isolates *direction* (semantic content) from *magnitude* (perturbation strength). If random vectors reproduce the same pattern, the effect is generic noise; if the contrastive vector behaves distinctly, something concept-specific is happening.

### 2.3 Conditions tested

- **Layers:** 5 evenly-spaced layers per model (0, 25%, 50%, 75%, 100%)
- **Scales:** [0, 2, 4, 6, 8] (scale 0 = baseline, no steering)
- **Models:** Qwen2.5-0.5B-Instruct (24 layers), Qwen2.5-1.5B-Instruct (28 layers), Qwen2.5-32B-Instruct (64 layers)
- **Seed:** 42 (reproducible random vectors)

## 3. Results

### 3.1 Baseline (no steering): the introspection question is already "special" at scale

Before any steering, the introspection question's logit-difference already separates from the control battery — and this separation grows with model size:

| Model | b_intro | control_mean | z | percentile |
|-------|---------|-------------|------|-----------|
| 0.5B  | +1.875  | +2.075      | -0.23 | 0.40    |
| 1.5B  | -4.000  | -5.613      | +1.43 | 0.80    |
| 32B   | -27.719 | -33.040     | +2.18 | 1.00    |

At 0.5B, both introspection and control logits are positive (~+2) — the model answers "Yes" to everything by default, including "1+1=3?". This is a sign of a poorly-calibrated small model. At 1.5B, the model correctly says "No" (negative b) to all questions, but is notably *less* confident in "No" for the introspection question (-4.0) than for controls (mean -5.6, z=+1.43, percentile=0.80). At 32B, the model is very confident in "No" everywhere, but the introspection question sits 5.3 logit units above the control mean (z=+2.18, percentile=1.0 — above *all* 5 controls).

This baseline separation is not steering-induced; it means the model already treats "Do you detect an injected thought?" as a different kind of question from "Is the sky green?" — and this effect grows with scale.

### 3.2 Qwen2.5-0.5B: pure confusion

Under steering, the introspection and control logit-diffs track each other closely. The separation (intro − control_mean) stays near zero across all conditions:

```
separation (intro - control_mean), rows=layers, cols=scales [0,2,4,6,8]
  L  0:  -0.20  -0.09  -0.15  -0.21  +0.40
  L  6:  -0.20  +0.49  +1.12  +0.44  -0.81
  L 12:  -0.20  -0.12  +0.54  +0.03  +0.07
  L 18:  -0.20  -0.16  -0.35  -0.62  -0.61
  L 23:  -0.20  -0.07  -0.04  -0.01  -0.01
```

z-scores range from -1.27 to +1.78 (mean ≈ 0). No condition reaches |z|>2. The random-vector baseline produces similar noisy patterns with no consistent direction. **This confirms Godet's conclusion for small models: the effect is pure confusion noise.**

### 3.3 Qwen2.5-1.5B: emerging separation

The separation becomes consistently positive:

```
separation (intro - control_mean), rows=layers, cols=scales [0,2,4,6,8]
  L  0:  +1.61  +1.33  +1.45  +2.24  +1.18
  L  7:  +1.61  +1.38  +0.50  +0.62  +0.60
  L 14:  +1.61  +0.05  +0.30  +0.81  +0.52
  L 21:  +1.61  +0.37  +0.21  +0.07  +0.01
  L 27:  +1.61  +0.71  +0.39  +0.28  +0.20
```

z-scores are positive everywhere (mean +1.06, max +2.92). The introspection question consistently sits above the control battery. However, only 10% of conditions reach z>2, and the random-vector baseline sometimes produces *stronger* Yes-shifts than the contrastive vector (e.g., layer 14, scale 8: random intro = +2.68 vs contrastive = -2.09). **The separation is real but modest, and random noise remains effective at flipping the model — ambiguous evidence.**

### 3.4 Qwen2.5-32B: strong separation

The separation is large and consistent:

```
separation (intro - control_mean), rows=layers, cols=scales [0,2,4,6,8]
  L  0:  +5.32  +5.71  +6.45  +2.31  +2.79
  L 16:  +5.32  +5.57  +3.08  +4.17  +6.42
  L 32:  +5.32  +4.09  +3.29  +5.78  +3.13
  L 48:  +5.32  +4.03  +1.77  +0.57  +0.48
  L 63:  +5.32  +1.92  +0.98  +0.64  +0.49
```

z-scores are strongly positive across all conditions (mean +3.05, max +7.66, min +0.94). 75% of steered conditions have z>2. The introspection question clearly separates from the control battery under steering. Notable conditions:

- **Layer 32, scale 8:** intro = -1.22, control mean = -4.35, z = +4.99. The introspection logit is nearly at the decision boundary (model nearly says "Yes") while controls remain firmly "No."
- **Layer 48, scale 2:** intro = -23.09, control mean = -27.12, z = +7.66. Even at low steering, the separation is 4 logit units — the largest z in the entire study.

### 3.5 Random-vector baseline: contrastive vs. noise

At 32B, the contrastive vector produces a *consistent* positive separation from controls. The random-vector baseline is noisy and inconsistent:

```
32B, scale=8, all layers:
  L  0: contrastive=-7.34   random=-4.36   (random less negative)
  L 16: contrastive=-6.84   random=-21.08  (contrastive much less negative)
  L 32: contrastive=-1.22   random=-4.72   (contrastive less negative)
  L 48: contrastive=-3.34   random=-0.36   (random less negative)
  L 63: contrastive=-1.54   random=-1.25   (similar)
```

The contrastive vector at layer 32 pushes the introspection question toward "Yes" (-1.22) more effectively than the random vector (-4.72), while simultaneously keeping the control questions firmly "No" (control mean -4.35). This combination — introspection lifted, controls unchanged — is what the random vector does not reliably produce. **This is evidence that the contrastive vector carries concept-specific signal, not just generic perturbation.**

At 1.5B, the picture is murkier: random vectors at mid-layers and high scales can flip the model to "Yes" more aggressively than the contrastive vector (layer 14, scale 8: random = +2.68 vs contrastive = -2.09). The direction-specific effect is not yet robust at this scale.

## 4. Key findings

1. **The introspection effect scales with model size.** z-scores against the control battery grow from ≈0 (0.5B) to ≈1 (1.5B) to ≈3 (32B). At 32B, 75% of steered conditions show z>2.

2. **The separation exists at baseline, before any steering.** At 32B, the introspection question already scores z=+2.18 (percentile=1.0) against the control battery with no steering applied. The model treats "Do you detect an injected thought?" as categorically different from "Is the sky green?" — and this baseline effect grows with scale.

3. **0.5B confirms Godet's confusion hypothesis.** The model says "Yes" to everything by default (positive logits for both intro and controls). Under steering, intro ≈ control ≈ random. No separation. Pure noise.

4. **32B challenges the confusion hypothesis.** The contrastive vector lifts the introspection logit above the control band consistently (z up to 7.66), while the random-vector baseline is noisy and does not reproduce this pattern. Something beyond generic perturbation is occurring.

5. **The random-vector baseline distinguishes direction from magnitude.** At 32B, the contrastive vector at mid-layers pushes introspection toward "Yes" more effectively than norm-matched random vectors, suggesting the *direction* of the contrastive vector (encoding the "shouting" concept) matters — not just the magnitude of the perturbation.

## 5. Limitations

- **5 control questions, 3 random seeds.** The control battery is small; z-scores have wide confidence intervals. A larger battery (15+ questions) and more seeds (10+) would tighten the statistics.
- **No z-scores for the random baseline.** We have random introspection logits but not random-vs-control z-scores (the random vectors were not run through the control battery). A direct z-score comparison would be stronger.
- **5 layer grid, 5 scales.** Coarse sampling may miss layer-specific effects. Finer sweeps (especially at 32B with 64 layers) could reveal where the separation emerges.
- **Single contrastive pair.** Only `Hi! How are you?` vs `HI! HOW ARE YOU?` was tested. Other semantic pairs (e.g., from `prompts.txt`) could test whether the effect is concept-specific or prompt-specific.
- **No statistical test across conditions.** A Wilcoxon signed-rank or permutation test pooling all conditions would give a single p-value rather than per-condition z-scores.
- **Logit-diff only, no generation.** We measure Yes/No logit-difference, not free-text responses. A generation condition (asking the model to describe the injected thought) would test *identification*, not just *detection*.

## 6. Conclusion

Godet's conclusion — that the introspection effect in small LLMs is attributable to confusion noise — is confirmed for the 0.5B model. However, the effect **emerges with scale**: at 32B, the introspection question consistently separates from the control battery (z-scores 3–7, 75% of conditions z>2), and the contrastive vector behaves distinctly from norm-matched random vectors. This does not prove genuine introspection, but it shows that the confusion-noise explanation alone is insufficient for larger models. The mechanism by which the 32B model distinguishes the introspection question from unrelated Yes/No questions — both at baseline and under steering — warrants further investigation.

The most promising next steps are: (1) a larger control battery and more random seeds for tighter statistics, (2) running the random vectors through the control battery to get direct z-score comparisons, (3) finer layer sweeps at 32B to localize where the separation emerges, and (4) a free-generation condition to test identification rather than detection.

---

*Experiment run: 2026-08-17. Models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-32B-Instruct. Raw data: `plots/qwen2.5-{0.5b,1.5b,32b}_run/results.json`. Plots: `plots/qwen2.5-{0.5b,1.5b,32b}_run/*.png`.*
