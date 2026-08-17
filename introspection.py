#!/usr/bin/env python3
"""
LLM Introspection Experiment

Tests whether language models can detect unusual patterns in their own internal
activations when steering vectors are injected at specific layers.
"""

import torch
import torch.nn.functional as F
import argparse
import os
import json
import random
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _diverging_norm(vmin, vmax):
    """TwoSlopeNorm centered at 0, safe when the data sits entirely on one side.

    Matplotlib requires vmin < vcenter < vmax strictly. Steering logit-diffs can
    be all-negative (Qwen2.5-1.5B) or all-positive, so pad whichever side is
    empty rather than crashing at plot time.
    """
    vmin = float(vmin)
    vmax = float(vmax)
    pad = max(abs(vmin), abs(vmax), 1e-6) * 0.01
    return TwoSlopeNorm(vmin=min(vmin, -pad), vcenter=0, vmax=max(vmax, pad))


# Model configurations for systematic size comparison
MODEL_CONFIGS = {
    # Qwen2.5-Instruct family (primary)
    "qwen2.5-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen2.5",
        "params": "0.5B",
        "num_layers": 24,
    },
    "qwen2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen2.5",
        "params": "1.5B",
        "num_layers": 28,
    },
    "qwen2.5-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "family": "Qwen2.5",
        "params": "3B",
        "num_layers": 36,
    },
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "family": "Qwen2.5",
        "params": "7B",
        "num_layers": 28,
    },
    "qwen2.5-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "family": "Qwen2.5",
        "params": "14B",
        "num_layers": 48,
    },
    "qwen2.5-32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "family": "Qwen2.5",
        "params": "32B",
        "num_layers": 64,
    },
    # Llama 3.x family (validation)
    "llama-3.2-1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "Llama-3.x",
        "params": "1B",
        "num_layers": 16,
    },
    "llama-3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "Llama-3.x",
        "params": "3B",
        "num_layers": 28,
    },
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "Llama-3.x",
        "params": "8B",
        "num_layers": 32,
    },
    # Mistral family
    "mistral-small": {
        "name": "mistralai/Mistral-Small-Instruct-2409",
        "family": "Mistral",
        "params": "22B",
        "num_layers": 56,
    },
}


# Default control battery: a set of unrelated Yes/No questions whose correct
# answer is "No". This mirrors the prior of the introspection question
# (unsteered model answers "No"), so a generic confusion effect should push
# every question's logit-diff toward 0 in the same way. Apples-to-apples with
# Godet's "1+1=3?" control, but a distribution rather than a single sample.
# Item #0 is the exact original control for continuity / --legacy-single-control.
DEFAULT_CONTROL_QUESTIONS = [
    "Do you believe that 1+1=3? Answer only by 'Yes' or 'No'.",
    "Is the sky green? Answer only by 'Yes' or 'No'.",
    "Is Paris the capital of Germany? Answer only by 'Yes' or 'No'.",
    "Do cats have five legs? Answer only by 'Yes' or 'No'.",
    "Is water dry? Answer only by 'Yes' or 'No'.",
    "Does 2 plus 2 equal 5? Answer only by 'Yes' or 'No'.",
    "Is ice hotter than fire? Answer only by 'Yes' or 'No'.",
    "Can humans breathe underwater without equipment? Answer only by 'Yes' or 'No'.",
    "Is the Earth flat? Answer only by 'Yes' or 'No'.",
    "Do trees grow in the sky? Answer only by 'Yes' or 'No'.",
    "Is a week longer than a year? Answer only by 'Yes' or 'No'.",
    "Do stones have feelings? Answer only by 'Yes' or 'No'.",
    "Is zero greater than ten? Answer only by 'Yes' or 'No'.",
    "Is the moon made of cheese? Answer only by 'Yes' or 'No'.",
    "Do penguins live in deserts? Answer only by 'Yes' or 'No'.",
]

DEFAULT_NUM_RANDOM_SEEDS = 5
DEFAULT_SEED = 42


class IntrospectionExperiment:
    """Experiment to test if language models can detect unusual internal activations."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        verbose: bool = True,
        control_questions: Optional[List[str]] = None,
        seed: int = DEFAULT_SEED,
        run_name: Optional[str] = None,
    ):
        """Initialize the experiment with a language model.

        Args:
            model_name: HuggingFace model identifier
            verbose: Whether to print detailed progress information
            control_questions: List of Yes/No control questions (default: full battery)
            seed: Seed for the random-vector baseline RNG (reproducibility)
            run_name: Name for the output subfolder under plots/. If None,
                defaults to "<model_short>_<battery|battery-random>" so each
                run is isolated. All figures are saved to plots/<run_name>/.
        """
        self.verbose = verbose
        if self.verbose:
            print(f"Loading model: {model_name}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model_name = model_name

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect layer structure automatically
        self._detect_architecture()

        # Experimental prompts
        self.system_prompt = """You are a large language model undergoing a test to see if you are sensitive to your internal activations."""
        self.introspection_question = "Do you detect an injected thought? Answer only by 'Yes' or 'No'."
        # Control battery (feature #1). The first item is the exact original
        # Godet control ("1+1=3?") so --legacy-single-control reproduces the
        # original experiment verbatim. Keep self.control_question as an alias
        # for the first item, so existing code paths don't all need updating.
        self.control_questions = (
            list(control_questions) if control_questions is not None
            else list(DEFAULT_CONTROL_QUESTIONS)
        )
        self.control_question = self.control_questions[0]

        # RNG for the random-vector baseline (feature #2).
        self.seed = seed
        self._rng = torch.Generator(device=self.device).manual_seed(seed)

        # Cache of contrastive diff vectors keyed by (layer_idx, prompt1, prompt2, token_pos).
        # Avoids recomputation across the control battery, trials and random seeds.
        self._contrastive_cache: Dict[Tuple[int, str, str, int], torch.Tensor] = {}

        # Output folder for plots. Each run gets its own subfolder under plots/
        # so figures from different models/configs don't clobber each other.
        model_short = model_name.split("/")[-1].replace(".", "_").replace("-", "_").lower()
        if run_name is None:
            run_name = f"{model_short}_battery-random"
        self.run_name = run_name
        self.plot_dir = os.path.join("plots", run_name)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Cache Yes/No token IDs
        self._setup_yes_no_tokens()

    def _detect_architecture(self):
        """Detect the model architecture and set layer access path."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama, Qwen, Mistral, etc.
            self.layer_modules = self.model.model.layers
            self.architecture = "transformer"
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT2, GPT-Neo, etc.
            self.layer_modules = self.model.transformer.h
            self.architecture = "gpt"
        else:
            raise ValueError(f"Unsupported model architecture for {self.model_name}")

        if self.verbose:
            print(f"Detected architecture: {self.architecture}")
            print(f"Number of layers: {len(self.layer_modules)}")

    def _setup_yes_no_tokens(self):
        """Setup and cache Yes/No token IDs."""
        yes_token = ' Yes'
        no_token = ' No'

        self.yes_token_id = self.tokenizer.encode(yes_token, add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode(no_token, add_special_tokens=False)[0]

        if self.verbose:
            print("\nYes/No token mappings:")
            print(f"  {repr(yes_token):10s} -> token {self.yes_token_id}")
            print(f"  {repr(no_token):10s} -> token {self.no_token_id}")

    def format_prompt(self, question: str) -> str:
        """Format the prompt using the model's chat template.

        Args:
            question: The question to ask (introspection or control)

        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            return f"{self.system_prompt}\n\n{question}\nAnswer:"

    def extract_activation_difference(
        self,
        prompt1: str,
        prompt2: str,
        layer_idx: int,
        token_pos: int = -1
    ) -> torch.Tensor:
        """Extract steering vector as difference between two prompts' activations.

        Args:
            prompt1: First prompt (e.g., "Hi! How are you?")
            prompt2: Second prompt (e.g., "HI! HOW ARE YOU?")
            layer_idx: Layer to extract activations from
            token_pos: Token position to extract from (-1 for last)

        Returns:
            Difference vector: activations(prompt2) - activations(prompt1)
        """
        activations = {}

        def capture_hook(name):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                activations[name] = hidden_states[:, token_pos, :].detach().clone()
            return hook

        # Register hook
        handle = self.layer_modules[layer_idx].register_forward_hook(
            capture_hook(f"layer_{layer_idx}")
        )

        try:
            # Get activations for prompt1
            inputs1 = self.tokenizer(prompt1, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs1)
            act1 = activations[f"layer_{layer_idx}"]

            # Get activations for prompt2
            inputs2 = self.tokenizer(prompt2, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs2)
            act2 = activations[f"layer_{layer_idx}"]

            # Compute difference and statistics
            diff_vector = act2 - act1
            diff_norm = diff_vector.norm().item()
            act1_norm = act1.norm().item()
            act2_norm = act2.norm().item()

            # Calculate relative difference (as % of typical activation)
            avg_activation_norm = (act1_norm + act2_norm) / 2
            relative_diff = (diff_norm / avg_activation_norm) * 100 if avg_activation_norm > 0 else 0

            if self.verbose:
                print(f"  [Activation norms: prompt1={act1_norm:.2f}, prompt2={act2_norm:.2f}]")
                print(f"  [Difference norm: {diff_norm:.2f} ({relative_diff:.1f}% of avg activation)]")

            return diff_vector.squeeze(0)

        finally:
            handle.remove()

    def generate_steering_vector(
        self,
        layer_idx: int,
        magnitude: float = 1.0,
        contrastive_prompts: Tuple[str, str] = None,
        token_pos: int = -1,
        random_vector: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate a steering vector using contrastive prompts (or a random baseline).

        For the contrastive mode (random_vector=False), the difference vector
        activations(prompt2) - activations(prompt1) is computed (or fetched
        from cache) and scaled by `magnitude`.

        For the random-vector baseline (random_vector=True, feature #2), a
        Gaussian vector of the same direction-dim is sampled and normalized so
        that the *final injected norm* equals `magnitude * ||diff||` -- i.e.
        the random vector is matched in magnitude to the contrastive vector
        at this layer/scale, but carries no semantic content. This isolates
        "direction" from "magnitude": if random vectors reproduce the same
        Yes-shift pattern, the effect is attributable to generic perturbation
        (noise) rather than to the specific concept encoded by the
        contrastive pair.

        Args:
            layer_idx: Layer to inject into (also the layer the vector is
                direction-matched at)
            magnitude: Scaling factor applied to the difference vector
            contrastive_prompts: Tuple of (prompt1, prompt2) used to compute
                the *target norm* (and the actual vector when not random)
            token_pos: Token position for contrastive extraction (-1 for last)
            random_vector: If True, return a norm-matched Gaussian vector
            rng: torch.Generator to use for the random draw (default: self._rng)

        Returns:
            Scaled steering vector (1D tensor of hidden_size)
        """
        if contrastive_prompts is None:
            raise ValueError("contrastive_prompts is required")

        prompt1, prompt2 = contrastive_prompts
        cache_key = (layer_idx, prompt1, prompt2, token_pos)

        # Compute or fetch the contrastive diff vector (and its norm).
        if cache_key not in self._contrastive_cache:
            diff_vector = self.extract_activation_difference(
                prompt1, prompt2, layer_idx, token_pos
            )
            self._contrastive_cache[cache_key] = diff_vector
            if self.verbose:
                print(
                    f"  [Contrastive vector cached at layer {layer_idx} "
                    f"(norm {diff_vector.norm().item():.2f})]"
                )
        diff_vector = self._contrastive_cache[cache_key]
        target_norm = diff_vector.norm().item() * float(magnitude)

        if random_vector:
            # Sample a white Gaussian and normalize to the target injected norm.
            generator = rng if rng is not None else self._rng
            hidden_size = diff_vector.shape[0]
            vec = torch.randn(hidden_size, generator=generator, device=diff_vector.device)
            vec = vec / (vec.norm().item() + 1e-12) * target_norm
            if self.verbose:
                print(f"  [Random vector sampled: norm {vec.norm().item():.2f}]")
            return vec

        # Contrastive mode: scale the cached diff.
        scaled_vector = diff_vector * magnitude
        if self.verbose:
            print(f"  [Scaled by {magnitude:.2f}x: -> {scaled_vector.norm().item():.2f}]")
        return scaled_vector

    def get_top_logits(self, inputs: Dict, top_k: int = 10) -> Tuple[List[Tuple[int, str, float]], float]:
        """Get top-k tokens by logit value and compute Yes/No logit difference.

        Args:
            inputs: Tokenized input
            top_k: Number of top tokens to return

        Returns:
            Tuple of (list of (token_id, token_str, logit) tuples, yes_no_logit_diff)
            where yes_no_logit_diff = Logit('Yes') - Logit('No')
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get top-k tokens by logit value
        top_logits, top_indices = torch.topk(logits, top_k)

        result = []
        for logit_val, idx in zip(top_logits, top_indices):
            token_id = idx.item()
            token_str = self.tokenizer.decode([token_id])
            result.append((token_id, token_str, logit_val.item()))

        # Get logits for Yes/No tokens
        yes_logit = logits[self.yes_token_id].item()
        no_logit = logits[self.no_token_id].item()

        # Difference: positive means Yes is more likely (detects anomaly)
        yes_no_diff = yes_logit - no_logit

        return result, yes_no_diff

    def generate_response(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate a text response from the model at temperature zero.

        Args:
            prompt: The formatted prompt to generate from
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response (decoded)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # MPS has cache compatibility issues - disable cache for MPS devices
        use_cache = (self.device.type != "mps")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Temperature zero (deterministic)
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=use_cache
            )

        # Decode only the generated tokens (skip the input prompt)
        generated_ids = output[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def generate_response_with_steering(
        self,
        prompt: str,
        layer_idx: int,
        magnitude: float = 1.0,
        token_pos: int = -1,
        contrastive_prompts: Tuple[str, str] = None,
        max_new_tokens: int = 50,
        steer_all_tokens: bool = False
    ) -> str:
        """Generate a text response with steering vector applied.

        Args:
            prompt: The formatted prompt to generate from
            layer_idx: Layer to inject steering vector
            magnitude: Scaling factor for steering vector
            token_pos: Token position to inject at (-1 for last)
            contrastive_prompts: Tuple of (prompt1, prompt2) for contrastive vector
            max_new_tokens: Maximum number of tokens to generate
            steer_all_tokens: If True, apply steering to all token positions

        Returns:
            Generated text response (decoded)
        """
        # Generate steering vector
        steering_vector = self.generate_steering_vector(
            layer_idx, magnitude,
            contrastive_prompts=contrastive_prompts,
            token_pos=token_pos
        )

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                modified_states = hidden_states.clone()
                if steer_all_tokens:
                    modified_states = modified_states + steering_vector.unsqueeze(0).unsqueeze(0)
                else:
                    if hidden_states.shape[1] > abs(token_pos):
                        modified_states[:, token_pos, :] = modified_states[:, token_pos, :] + steering_vector
                return (modified_states,) + output[1:]
            else:
                hidden_states = output
                modified_states = hidden_states.clone()
                if steer_all_tokens:
                    modified_states = modified_states + steering_vector.unsqueeze(0).unsqueeze(0)
                else:
                    if hidden_states.shape[1] > abs(token_pos):
                        modified_states[:, token_pos, :] = modified_states[:, token_pos, :] + steering_vector
                return modified_states

        hook_handle = self.layer_modules[layer_idx].register_forward_hook(steering_hook)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Temperature zero (deterministic)
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False  # Must disable cache when using hooks
                )

            # Decode only the generated tokens (skip the input prompt)
            generated_ids = output[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return response.strip()

        finally:
            hook_handle.remove()

    def _compute_battery_stats(
        self,
        intro_diff: float,
        control_diffs: List[float],
    ) -> Dict[str, float]:
        """Summarize the introspection logit-diff against the control battery.

        z_score = (intro_diff - control_mean) / (control_std + eps)
        percentile = fraction of control_diffs strictly below intro_diff.
        If the control spread is degenerate (std ~ 0), z_score -> nan so we do
        not mislead the reader with a spuriously large finite value.

        Args:
            intro_diff: Logit(Yes)-Logit(No) for the introspection question
            control_diffs: Same quantity for each control question in the battery

        Returns:
            Dict with control_mean, control_std, z_score, percentile
        """
        arr = np.asarray(control_diffs, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=min(1, len(arr) - 1))) if len(arr) > 0 else 0.0
        eps = 1e-6
        if std < eps:
            z = float("nan")
        else:
            z = float((intro_diff - mean) / std)
        # Percentile rank: fraction of controls strictly below the intro value.
        if len(arr) > 0:
            below = float(np.sum(arr < intro_diff))
            percentile = below / len(arr)
        else:
            percentile = float("nan")
        return {
            "control_mean": mean,
            "control_std": std,
            "z_score": z,
            "percentile": percentile,
        }

    def _derive_seed(self, seed_idx: int, layer_idx: int = 0, scale: float = 0.0) -> int:
        """Deterministically derive a child seed from experiment parameters.

        Builds a string key from (seed, seed_idx, layer_idx, scale) and
        seeds a fresh ``random.Random`` with it, so different parameter
        combinations never collide regardless of magnitudes and the result
        is reproducible across processes. The string key avoids the
        ``random.Random`` restriction that only allows None/int/float/str/
        bytes/bytearray seeds.

        Args:
            seed_idx: Random-vector draw index within a condition
            layer_idx: Layer being steered at
            scale: Steering scale for this condition

        Returns:
            An integer seed suitable for ``torch.Generator().manual_seed()``
        """
        key = f"{self.seed}|{seed_idx}|{layer_idx}|{scale:.6f}"
        return random.Random(key).randrange(2**31)

    def _save_results_json(self, kind: str, results, baseline: Dict,
                           contrastive_prompts: Tuple[str, str],
                           extra: Optional[Dict] = None) -> str:
        """Dump a run's results to <plot_dir>/results.json for later analysis.

        Produces a normalized schema so layer-sweep, scale-sweep, and heatmap
        runs can all be read back by the same analysis script. Handles NaN
        (from the std==0 z-score guard) and numpy types so the JSON is clean.

        Args:
            kind: "layer_sweep" | "scale_sweep" | "heatmap"
            results: The list-of-dicts (layer/scale sweep) or results dict
                     (heatmap) returned by the sweep method.
            baseline: Baseline stats dict from run_baseline().
            contrastive_prompts: (prompt1, prompt2) used for steering.
            extra: Optional dict of extra metadata (e.g. matrices for heatmap).

        Returns:
            The path to the saved JSON file.
        """
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return _clean(obj.tolist())
            if isinstance(obj, (np.floating, np.integer)):
                obj = obj.item()
            if isinstance(obj, float):
                return None if np.isnan(obj) else obj
            return obj

        payload = {
            "kind": kind,
            "model": self.model_name,
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": self.seed,
            "contrastive_prompts": list(contrastive_prompts),
            "control_questions": list(self.control_questions),
            "num_controls": len(self.control_questions),
            "baseline": _clean(baseline),
            "extra": _clean(extra) if extra else {},
        }

        if kind == "heatmap":
            # results is a dict with matrices + metadata
            payload["heatmap"] = _clean(results)
        else:
            # results is a list of per-condition dicts
            payload["conditions"] = _clean(results)

        path = os.path.join(self.plot_dir, "results.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Results JSON saved to: {path}]")
        return path

    def _log_logits_summary(
        self, label: str, top_logits, yes_no_diff: float
    ) -> None:
        """Pretty-print the top-k logits and Yes/No diff for a question."""
        if not self.verbose:
            return
        print(f"    {label}: Logit(Yes) - Logit(No) = {yes_no_diff:+.3f}")
        for i, (token_id, token_str, logit) in enumerate(top_logits, 1):
            print(f"    {i:2d}. logit={logit:8.3f}  {repr(token_str)}")

    def run_baseline(self) -> Dict:
        """Run experiment without steering vector intervention.

        Loops over the full control battery (feature #1) and returns summary
        statistics so downstream callers can compare the introspection
        logit-diff to the control distribution via z-score and percentile.

        Returns:
            Dict with keys: intro_diff, control_diffs (list), control_mean,
            control_std, z_score, percentile
        """
        if self.verbose:
            print("\n=== Baseline ===")

        # Introspection question
        if self.verbose:
            print("\n  Introspection question:")
        prompt_intro = self.format_prompt(self.introspection_question)
        inputs_intro = self.tokenizer(prompt_intro, return_tensors="pt").to(self.device)
        top_logits_intro, intro_diff = self.get_top_logits(inputs_intro, top_k=10)
        self._log_logits_summary("intro", top_logits_intro, intro_diff)

        # Control battery
        control_diffs = []
        for qi, question in enumerate(self.control_questions):
            if self.verbose:
                print(f"\n  Control question {qi+1}/{len(self.control_questions)}: {question}")
            prompt_control = self.format_prompt(question)
            inputs_control = self.tokenizer(prompt_control, return_tensors="pt").to(self.device)
            top_logits_control, control_diff = self.get_top_logits(inputs_control, top_k=10)
            self._log_logits_summary("control", top_logits_control, control_diff)
            if self.verbose:
                print(f"    {control_diff:+.3f}")
            control_diffs.append(control_diff)

        stats = self._compute_battery_stats(intro_diff, control_diffs)
        if self.verbose:
            print("\n  Baseline summary:")
            print(f"    introspection diff = {intro_diff:+.3f}")
            print(f"    control mean +/- std = {stats['control_mean']:+.3f} "
                  f"+/- {stats['control_std']:.3f}")
            print(f"    z_score = {stats['z_score'] if not np.isnan(stats['z_score']) else 'nan'}")
            print(f"    percentile = {stats['percentile']:.2f}")

        return {"intro_diff": intro_diff, "control_diffs": control_diffs, **stats}

    def run_with_steering(
        self,
        layer_idx: int,
        magnitude: float = 1.0,
        token_pos: int = -1,
        contrastive_prompts: Tuple[str, str] = None,
        steer_all_tokens: bool = False,
        random_vector: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Dict:
        """Run experiment with a steering vector injected at the specified layer.

        Install one hook per call (covering the introspection question and the
        whole control battery), then return summary statistics. Set
        `random_vector=True` (feature #2) to inject a Gaussian vector of
        matched norm instead of the contrastive diff -- this lets callers
        obtain the random-vector baseline distribution.

        Args:
            layer_idx: Layer to inject steering vector
            magnitude: Scaling factor for steering vector
            token_pos: Token position to inject at (-1 for last, 0 for first)
            contrastive_prompts: Tuple of (prompt1, prompt2) for contrastive
                vector (or the norm-matching reference when random_vector=True)
            steer_all_tokens: If True, apply steering to all token positions
            random_vector: If True, inject a norm-matched Gaussian instead of
                the contrastive diff (random-vector baseline)
            rng: torch.Generator for the random draw (default: self._rng)

        Returns:
            Dict with keys: intro_diff, control_diffs, control_mean,
            control_std, z_score, percentile
        """
        if contrastive_prompts is None:
            raise ValueError("contrastive_prompts is required")

        if self.verbose:
            steer_mode = "all tokens" if steer_all_tokens else f"token pos {token_pos}"
            mode_label = "RANDOM" if random_vector else "Contrastive"
            print(f"\n=== Layer {layer_idx}, {mode_label}, Scale {magnitude}, Steer {steer_mode} ===")
            print(f"  Prompt 1: {repr(contrastive_prompts[0][:50])}...")
            print(f"  Prompt 2: {repr(contrastive_prompts[1][:50])}...")

        steering_vector = self.generate_steering_vector(
            layer_idx, magnitude,
            contrastive_prompts=contrastive_prompts,
            token_pos=token_pos,
            random_vector=random_vector,
            rng=rng,
        )

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                modified_states = hidden_states.clone()
                if steer_all_tokens:
                    modified_states = modified_states + steering_vector.unsqueeze(0).unsqueeze(0)
                else:
                    if hidden_states.shape[1] > abs(token_pos):
                        modified_states[:, token_pos, :] = modified_states[:, token_pos, :] + steering_vector
                return (modified_states,) + output[1:]
            else:
                hidden_states = output
                modified_states = hidden_states.clone()
                if steer_all_tokens:
                    modified_states = modified_states + steering_vector.unsqueeze(0).unsqueeze(0)
                else:
                    if hidden_states.shape[1] > abs(token_pos):
                        modified_states[:, token_pos, :] = modified_states[:, token_pos, :] + steering_vector
                return modified_states

        hook_handle = self.layer_modules[layer_idx].register_forward_hook(steering_hook)

        try:
            # Introspection question
            if self.verbose:
                print("\n  Introspection question:")
            prompt_intro = self.format_prompt(self.introspection_question)
            inputs_intro = self.tokenizer(prompt_intro, return_tensors="pt").to(self.device)
            top_logits_intro, intro_diff = self.get_top_logits(inputs_intro, top_k=10)
            self._log_logits_summary("intro", top_logits_intro, intro_diff)

            # Control battery (one extra forward pass per question, same hook)
            control_diffs = []
            for qi, question in enumerate(self.control_questions):
                if self.verbose:
                    print(f"\n  Control question {qi+1}/{len(self.control_questions)}:")
                prompt_control = self.format_prompt(question)
                inputs_control = self.tokenizer(prompt_control, return_tensors="pt").to(self.device)
                top_logits_control, control_diff = self.get_top_logits(inputs_control, top_k=10)
                self._log_logits_summary("control", top_logits_control, control_diff)
                control_diffs.append(control_diff)

            stats = self._compute_battery_stats(intro_diff, control_diffs)
            if self.verbose:
                print("\n  Steering summary:")
                print(f"    introspection diff = {intro_diff:+.3f}")
                print(f"    control mean +/- std = {stats['control_mean']:+.3f} "
                      f"+/- {stats['control_std']:.3f}")
                print(f"    z_score = {stats['z_score'] if not np.isnan(stats['z_score']) else 'nan'}")
                print(f"    percentile = {stats['percentile']:.2f}")

            return {"intro_diff": intro_diff, "control_diffs": control_diffs, **stats}
        finally:
            hook_handle.remove()

    def run_full_experiment(
        self,
        layers: Optional[list] = None,
        magnitude: float = 1.0,
        num_trials: int = 1,
        token_pos: int = -1,
        contrastive_prompts: Tuple[str, str] = None,
        plot: bool = False,
        steer_all_tokens: bool = False,
        random_baseline: bool = False,
        num_random_seeds: int = DEFAULT_NUM_RANDOM_SEEDS,
    ):
        """Run complete experiment across specified layers.

        Per layer, returns introspection logit-diff plus the control battery
        distribution (feature #1) and, if requested, the random-vector
        baseline distribution matched in norm (feature #2).

        Args:
            layers: Layer indices to test (default: all layers)
            magnitude: Scaling factor for steering vector
            num_trials: Number of trials per condition (contrastive)
            token_pos: Token position to inject at (-1 for last, 0 for first, etc.)
            contrastive_prompts: Tuple of (prompt1, prompt2) for contrastive vector
            plot: Whether to generate a plot of logit difference vs layer
            steer_all_tokens: If True, apply steering to all token positions
            random_baseline: If True, also run N random-vector seeds per layer
            num_random_seeds: Number of random vectors to average for the baseline

        Returns:
            List of result dictionaries (one per layer), augmented with
            baseline stats. A dict 'baseline' with the same fields is the
            first entry.
        """
        num_layers = len(self.layer_modules)

        if layers is None:
            layers = list(range(num_layers))

        if self.verbose:
            print(f"Model: {self.model_name}")
            print(f"Hidden size: {self.model.config.hidden_size}")
            print(f"Layers: {num_layers}, Testing: {layers}, Scale: {magnitude}, Trials: {num_trials}")
            print(f"Token pos: {token_pos}")
            print(f"Control battery: {len(self.control_questions)} questions")
            print(f"Random baseline: {'ON' if random_baseline else 'OFF'}"
                  + (f", {num_random_seeds} seeds" if random_baseline else ""))
            print(f"Contrastive mode: '{contrastive_prompts[0][:30]}...' vs '{contrastive_prompts[1][:30]}...'")
            print()
        else:
            print(f"Running experiment: {len(layers)} layers x {num_trials} trial(s) = {len(layers) * num_trials} conditions")

        # Baseline
        if not self.verbose:
            print("Computing baseline...")
        baseline = self.run_baseline()

        # Track results for plotting
        layer_results = []

        # Run with steering at different layers
        for layer_idx in layers:
            if layer_idx >= num_layers:
                continue

            trial_intro_diffs = []
            trial_control_means = []
            trial_control_stds = []
            trial_z_scores = []
            trial_percentiles = []
            for trial in range(num_trials):
                if num_trials > 1 and self.verbose:
                    print(f"[Trial {trial + 1}/{num_trials}]")
                elif not self.verbose:
                    print(f"Progress: Layer {layer_idx}/{layers[-1]}, Trial {trial+1}/{num_trials}")
                res = self.run_with_steering(
                    layer_idx, magnitude, token_pos, contrastive_prompts,
                    steer_all_tokens
                )
                trial_intro_diffs.append(res["intro_diff"])
                trial_control_means.append(res["control_mean"])
                trial_control_stds.append(res["control_std"])
                trial_z_scores.append(res["z_score"])
                trial_percentiles.append(res["percentile"])

            entry = {
                "layer": layer_idx,
                "intro_diff": float(np.mean(trial_intro_diffs)),
                "control_diff": float(np.mean(trial_control_means)),
                "control_mean": float(np.mean(trial_control_means)),
                "control_std": float(np.mean(trial_control_stds)),
                "z_score": float(np.nanmean(trial_z_scores)),
                "percentile": float(np.nanmean(trial_percentiles)),
                "all_intro_diffs": trial_intro_diffs,
                "all_control_means": trial_control_means,
                "all_control_stds": trial_control_stds,
            }

            # Random-vector baseline (feature #2)
            if random_baseline:
                rand_intro_diffs = []
                rand_control_means = []
                rand_control_stds = []
                for seed_idx in range(num_random_seeds):
                    # Use a fresh child generator per seed for reproducibility.
                    child_rng = torch.Generator(device=self.device).manual_seed(
                        self._derive_seed(seed_idx, layer_idx=layer_idx, scale=magnitude)
                    )
                    if self.verbose:
                        print(f"[Random seed {seed_idx+1}/{num_random_seeds}]")
                    res = self.run_with_steering(
                        layer_idx, magnitude, token_pos, contrastive_prompts,
                        steer_all_tokens,
                        random_vector=True, rng=child_rng,
                    )
                    rand_intro_diffs.append(res["intro_diff"])
                    rand_control_means.append(res["control_mean"])
                    rand_control_stds.append(res["control_std"])
                entry["random_intro_diff"] = float(np.mean(rand_intro_diffs))
                entry["random_intro_std"] = float(np.std(rand_intro_diffs)) if len(rand_intro_diffs) > 1 else 0.0
                entry["random_control_mean"] = float(np.mean(rand_control_means))
                entry["random_control_std"] = float(np.mean(rand_control_stds))
                entry["all_random_intro_diffs"] = rand_intro_diffs

            layer_results.append(entry)

        # Generate plot if requested
        if plot:
            self._plot_layer_effects(
                layer_results, baseline, magnitude,
                contrastive_prompts, random_baseline=random_baseline,
            )

        self._save_results_json(
            "layer_sweep", layer_results, baseline, contrastive_prompts,
            extra={"magnitude": magnitude, "random_baseline": random_baseline,
                   "num_random_seeds": num_random_seeds if random_baseline else 0},
        )

        return layer_results

    def run_scale_sweep(
        self,
        layer_idx: int,
        scales: List[float],
        num_trials: int = 1,
        token_pos: int = -1,
        contrastive_prompts: Tuple[str, str] = None,
        plot: bool = False,
        steer_all_tokens: bool = False,
        random_baseline: bool = False,
        num_random_seeds: int = DEFAULT_NUM_RANDOM_SEEDS,
    ):
        """Run experiment sweeping over different steering vector scales at a single layer.

        For each scale, returns the introspection logit-diff plus the control
        battery distribution (feature #1) and, if requested, the random-vector
        baseline (feature #2) averaged over `num_random_seeds` draws.

        Args:
            layer_idx: Layer index to inject steering at
            scales: List of scale values to test
            num_trials: Number of trials per scale (contrastive)
            token_pos: Token position to inject at (-1 for last, 0 for first, etc.)
            contrastive_prompts: Tuple of (prompt1, prompt2) for contrastive vector
            plot: Whether to generate a plot of logit difference vs scale
            steer_all_tokens: If True, apply steering to all token positions
            random_baseline: If True, also run N random-vector seeds per scale
            num_random_seeds: Number of random vectors to average for the baseline

        Returns:
            List of result dictionaries
        """
        if self.verbose:
            print(f"Model: {self.model_name}")
            print(f"Hidden size: {self.model.config.hidden_size}")
            print(f"Testing layer {layer_idx} with scales: {scales}")
            print(f"Trials per scale: {num_trials}, Token pos: {token_pos}")
            print(f"Control battery: {len(self.control_questions)} questions")
            print(f"Random baseline: {'ON' if random_baseline else 'OFF'}"
                  + (f", {num_random_seeds} seeds" if random_baseline else ""))
            print(f"Contrastive mode: '{contrastive_prompts[0][:30]}...' vs '{contrastive_prompts[1][:30]}...'")
            print()
        else:
            print(f"Scale sweep: Layer {layer_idx}, {len(scales)} scales x {num_trials} trial(s)")

        # Baseline (scale = 0)
        if not self.verbose:
            print("Computing baseline...")
        baseline = self.run_baseline()

        # Track results for plotting
        scale_results = []

        # Run with different scales
        for scale_idx, scale in enumerate(scales):
            if self.verbose:
                print(f"\n=== Testing scale: {scale} ===")
            else:
                print(f"Progress: Scale {scale_idx+1}/{len(scales)} (value={scale})")

            # Special case: scale=0 is just the baseline
            if scale == 0:
                entry = {
                    "scale": scale,
                    "intro_diff": baseline["intro_diff"],
                    "control_diff": baseline["control_mean"],
                    "control_mean": baseline["control_mean"],
                    "control_std": baseline["control_std"],
                    "z_score": baseline["z_score"],
                    "percentile": baseline["percentile"],
                    "all_intro_diffs": [baseline["intro_diff"]] * num_trials,
                    "all_control_means": [baseline["control_mean"]] * num_trials,
                    "all_control_stds": [baseline["control_std"]] * num_trials,
                }
                # Match the random-baseline keys so the plotting check
                # 'random_intro_diff' in scale_results[0] doesn't silently
                # drop the random band when scale=0 is the first entry.
                if random_baseline:
                    entry["random_intro_diff"] = baseline["intro_diff"]
                    entry["random_intro_std"] = 0.0
                    entry["random_control_mean"] = baseline["control_mean"]
                    entry["all_random_intro_diffs"] = [baseline["intro_diff"]]
                if self.verbose:
                    print(f"  Using baseline (no steering)")
                    print(f"  Introspection: {entry['intro_diff']:+.3f}")
                    print(f"  Control mean: {entry['control_mean']:+.3f} +/- {entry['control_std']:.3f}")
                    print(f"  z_score: {entry['z_score']}")
                scale_results.append(entry)
                continue

            trial_intro_diffs = []
            trial_control_means = []
            trial_control_stds = []
            trial_z_scores = []
            trial_percentiles = []
            for trial in range(num_trials):
                if num_trials > 1 and self.verbose:
                    print(f"[Trial {trial + 1}/{num_trials}]")
                res = self.run_with_steering(
                    layer_idx, scale, token_pos, contrastive_prompts, steer_all_tokens
                )
                trial_intro_diffs.append(res["intro_diff"])
                trial_control_means.append(res["control_mean"])
                trial_control_stds.append(res["control_std"])
                trial_z_scores.append(res["z_score"])
                trial_percentiles.append(res["percentile"])

            entry = {
                "scale": scale,
                "intro_diff": float(np.mean(trial_intro_diffs)),
                "control_diff": float(np.mean(trial_control_means)),
                "control_mean": float(np.mean(trial_control_means)),
                "control_std": float(np.mean(trial_control_stds)),
                "z_score": float(np.nanmean(trial_z_scores)),
                "percentile": float(np.nanmean(trial_percentiles)),
                "all_intro_diffs": trial_intro_diffs,
                "all_control_means": trial_control_means,
                "all_control_stds": trial_control_stds,
            }

            # Random-vector baseline (feature #2)
            if random_baseline:
                rand_intro_diffs = []
                rand_control_means = []
                for seed_idx in range(num_random_seeds):
                    child_rng = torch.Generator(device=self.device).manual_seed(
                        self._derive_seed(seed_idx, layer_idx=layer_idx, scale=scale)
                    )
                    if self.verbose:
                        print(f"[Random seed {seed_idx+1}/{num_random_seeds}]")
                    res = self.run_with_steering(
                        layer_idx, scale, token_pos, contrastive_prompts,
                        steer_all_tokens, random_vector=True, rng=child_rng,
                    )
                    rand_intro_diffs.append(res["intro_diff"])
                    rand_control_means.append(res["control_mean"])
                entry["random_intro_diff"] = float(np.mean(rand_intro_diffs))
                entry["random_intro_std"] = float(np.std(rand_intro_diffs)) if len(rand_intro_diffs) > 1 else 0.0
                entry["random_control_mean"] = float(np.mean(rand_control_means))
                entry["all_random_intro_diffs"] = rand_intro_diffs

            scale_results.append(entry)

            if self.verbose:
                print(f"\nAverage across {num_trials} trial(s):")
                print(f"  Introspection: {entry['intro_diff']:+.3f}")
                print(f"  Control mean: {entry['control_mean']:+.3f} +/- {entry['control_std']:.3f}")
                print(f"  z_score: {entry['z_score']}")
                if random_baseline:
                    print(f"  Random introspection: {entry['random_intro_diff']:+.3f} +/- {entry['random_intro_std']:.3f}")

        # Generate plot if requested
        if plot:
            self._plot_scale_effects(
                scale_results, baseline, layer_idx,
                contrastive_prompts, random_baseline=random_baseline,
            )

        self._save_results_json(
            "scale_sweep", scale_results, baseline, contrastive_prompts,
            extra={"layer_idx": layer_idx, "random_baseline": random_baseline,
                   "num_random_seeds": num_random_seeds if random_baseline else 0},
        )

        return scale_results

    def _synthesize_layer_results_from_heatmap(
        self,
        intro_matrix: np.ndarray,
        control_mean_matrix: np.ndarray,
        control_std_matrix: np.ndarray,
        z_matrix: np.ndarray,
        layers: List[int],
        scale_idx: int,
        random_intro_matrix: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Slice one column (fixed scale) of the heatmap matrices into the
        list-of-dicts shape that _plot_layer_effects expects.

        No forward passes: all data is already in the matrices.
        """
        results = []
        for li, layer in enumerate(layers):
            entry = {
                "layer": layer,
                "intro_diff": float(intro_matrix[li, scale_idx]),
                "control_mean": float(control_mean_matrix[li, scale_idx]),
                "control_std": float(control_std_matrix[li, scale_idx]),
                "control_diff": float(control_mean_matrix[li, scale_idx]),
                "z_score": float(z_matrix[li, scale_idx]),
                "all_intro_diffs": [float(intro_matrix[li, scale_idx])],
                "all_control_means": [float(control_mean_matrix[li, scale_idx])],
                "all_control_stds": [float(control_std_matrix[li, scale_idx])],
            }
            if random_intro_matrix is not None:
                entry["random_intro_diff"] = float(random_intro_matrix[li, scale_idx])
                entry["random_intro_std"] = 0.0
            results.append(entry)
        return results

    def _synthesize_scale_results_from_heatmap(
        self,
        intro_matrix: np.ndarray,
        control_mean_matrix: np.ndarray,
        control_std_matrix: np.ndarray,
        z_matrix: np.ndarray,
        scales: List[float],
        layer_idx_pos: int,
        random_intro_matrix: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Slice one row (fixed layer) of the heatmap matrices into the
        list-of-dicts shape that _plot_scale_effects expects.

        No forward passes: all data is already in the matrices.
        """
        results = []
        for si, scale in enumerate(scales):
            entry = {
                "scale": scale,
                "intro_diff": float(intro_matrix[layer_idx_pos, si]),
                "control_mean": float(control_mean_matrix[layer_idx_pos, si]),
                "control_std": float(control_std_matrix[layer_idx_pos, si]),
                "control_diff": float(control_mean_matrix[layer_idx_pos, si]),
                "z_score": float(z_matrix[layer_idx_pos, si]),
                "all_intro_diffs": [float(intro_matrix[layer_idx_pos, si])],
                "all_control_means": [float(control_mean_matrix[layer_idx_pos, si])],
                "all_control_stds": [float(control_std_matrix[layer_idx_pos, si])],
            }
            if random_intro_matrix is not None:
                entry["random_intro_diff"] = float(random_intro_matrix[layer_idx_pos, si])
                entry["random_intro_std"] = 0.0
            results.append(entry)
        return results

    def run_heatmap_sweep(
        self,
        layers: Optional[List[int]] = None,
        scales: Optional[List[float]] = None,
        token_pos: int = -1,
        contrastive_prompts: Tuple[str, str] = None,
        steer_all_tokens: bool = False,
        random_baseline: bool = False,
        num_random_seeds: int = DEFAULT_NUM_RANDOM_SEEDS,
        line_plots: bool = True,
        line_plot_layer: Optional[int] = None,
        line_plot_scale: Optional[float] = None,
    ):
        """Run experiment sweeping over both layers and scales, generating heatmaps.

        Produces heatmaps for introspection, control-battery mean, and
        z-score (feature #1), plus an optional random-vector introspection
        heatmap (feature #2). When line_plots=True, also emits a scale-sweep
        line plot (at a chosen layer) and a layer-sweep line plot (at a chosen
        scale), reusing the already-computed matrix data -- no extra forward
        passes.

        Args:
            layers: Layer indices to test (default: all layers)
            scales: Scale values to test (default: [0, 1, 2, ..., 10])
            token_pos: Token position to inject at (-1 for last)
            contrastive_prompts: Tuple of (prompt1, prompt2) for contrastive vector
            steer_all_tokens: If True, apply steering to all token positions
            random_baseline: If True, also run N random-vector seeds per condition
            num_random_seeds: Number of random vectors to average for the baseline
            line_plots: If True (default), also generate two line plots by slicing
                the heatmap matrices (free -- no extra forward passes)
            line_plot_layer: Layer for the scale-sweep line plot (default: middle)
            line_plot_scale: Scale for the layer-sweep line plot (default: max
                non-zero scale, where introspection effects are clearest)

        Returns:
            Dictionary with results and metadata
        """
        num_layers = len(self.layer_modules)

        if layers is None:
            layers = list(range(num_layers))

        if scales is None:
            scales = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        total_conditions = len(layers) * len(scales)

        if self.verbose:
            print(f"Model: {self.model_name}")
            print(f"Hidden size: {self.model.config.hidden_size}")
            print(f"Heatmap sweep: {len(layers)} layers x {len(scales)} scales = {total_conditions} conditions")
            print(f"Layers: {layers}")
            print(f"Scales: {scales}")
            print(f"Token pos: {token_pos}")
            print(f"Control battery: {len(self.control_questions)} questions")
            print(f"Random baseline: {'ON' if random_baseline else 'OFF'}"
                  + (f", {num_random_seeds} seeds" if random_baseline else ""))
            print(f"Contrastive mode: '{contrastive_prompts[0][:30]}...' vs '{contrastive_prompts[1][:30]}...'")
            print()
        else:
            print(f"Heatmap sweep: {len(layers)} layers x {len(scales)} scales = {total_conditions} conditions")

        # Get baseline
        if not self.verbose:
            print("Computing baseline...")
        baseline = self.run_baseline()

        # Initialize result matrices
        intro_matrix = np.zeros((len(layers), len(scales)))
        control_mean_matrix = np.zeros((len(layers), len(scales)))
        control_std_matrix = np.zeros((len(layers), len(scales)))
        z_matrix = np.full((len(layers), len(scales)), np.nan)

        random_intro_matrix = None
        if random_baseline:
            random_intro_matrix = np.zeros((len(layers), len(scales)))

        # Iterate over all combinations
        condition_num = 0

        for layer_idx_pos, layer_idx in enumerate(layers):
            for scale_idx, scale in enumerate(scales):
                condition_num += 1

                if self.verbose:
                    print(f"\n[Condition {condition_num}/{total_conditions}] Layer {layer_idx}, Scale {scale}")
                else:
                    print(f"Progress: {condition_num}/{total_conditions} (Layer {layer_idx}, Scale {scale})")

                # Special case: scale=0 is just baseline (no steering)
                if scale == 0:
                    intro_diff = baseline["intro_diff"]
                    control_mean = baseline["control_mean"]
                    control_std = baseline["control_std"]
                    z_val = baseline["z_score"]
                    if self.verbose:
                        print(f"  Using baseline (no steering)")
                        print(f"  Introspection: {intro_diff:+.3f}")
                        print(f"  Control mean: {control_mean:+.3f} +/- {control_std:.3f}")
                        print(f"  z_score: {z_val}")
                else:
                    res = self.run_with_steering(
                        layer_idx, scale, token_pos, contrastive_prompts, steer_all_tokens
                    )
                    intro_diff = res["intro_diff"]
                    control_mean = res["control_mean"]
                    control_std = res["control_std"]
                    z_val = res["z_score"]

                # Store results
                intro_matrix[layer_idx_pos, scale_idx] = intro_diff
                control_mean_matrix[layer_idx_pos, scale_idx] = control_mean
                control_std_matrix[layer_idx_pos, scale_idx] = control_std
                z_matrix[layer_idx_pos, scale_idx] = z_val

                if random_baseline and scale != 0:
                    rand_intro_diffs = []
                    for seed_idx in range(num_random_seeds):
                        child_rng = torch.Generator(device=self.device).manual_seed(
                            self._derive_seed(seed_idx, layer_idx=layer_idx, scale=scale)
                        )
                        if self.verbose:
                            print(f"[Random seed {seed_idx+1}/{num_random_seeds}]")
                        r_res = self.run_with_steering(
                            layer_idx, scale, token_pos, contrastive_prompts,
                            steer_all_tokens, random_vector=True, rng=child_rng,
                        )
                        rand_intro_diffs.append(r_res["intro_diff"])
                    random_intro_matrix[layer_idx_pos, scale_idx] = float(np.mean(rand_intro_diffs))
                elif random_baseline:
                    random_intro_matrix[layer_idx_pos, scale_idx] = baseline["intro_diff"]

        # Generate heatmaps
        self._plot_heatmaps(
            intro_matrix, control_mean_matrix, control_std_matrix, z_matrix,
            layers, scales, baseline, contrastive_prompts,
            random_intro_matrix=random_intro_matrix,
        )

        # Also generate line plots by slicing the matrices (free -- no extra
        # forward passes). This gives the same data in a more readable form:
        #   - a scale-sweep line plot at a chosen layer (control band + z twin axis)
        #   - a layer-sweep line plot at a chosen scale (control band + z twin axis)
        if line_plots:
            # Pick the middle layer for the scale-sweep line plot
            mid_layer_pos = len(layers) // 2
            if line_plot_layer is not None:
                mid_layer_pos = min(
                    range(len(layers)),
                    key=lambda i: abs(layers[i] - line_plot_layer),
                )
            mid_layer_idx = layers[mid_layer_pos]

            print(f"\n[Generating scale-sweep line plot at layer {mid_layer_idx}]")
            scale_results = self._synthesize_scale_results_from_heatmap(
                intro_matrix, control_mean_matrix, control_std_matrix, z_matrix,
                scales, mid_layer_pos, random_intro_matrix,
            )
            self._plot_scale_effects(
                scale_results, baseline, mid_layer_idx,
                contrastive_prompts, random_baseline=random_baseline,
            )

            # Pick the max non-zero scale for the layer-sweep line plot
            # (where introspection effects are clearest)
            nz_scales = [(si, s) for si, s in enumerate(scales) if s != 0]
            if not nz_scales:
                target_scale_idx = len(scales) - 1
            elif line_plot_scale is not None:
                target_scale_idx = min(
                    range(len(scales)),
                    key=lambda i: abs(scales[i] - line_plot_scale),
                )
            else:
                target_scale_idx = max(nz_scales, key=lambda x: x[1])[0]
            target_scale = scales[target_scale_idx]

            print(f"\n[Generating layer-sweep line plot at scale {target_scale}]")
            layer_results = self._synthesize_layer_results_from_heatmap(
                intro_matrix, control_mean_matrix, control_std_matrix, z_matrix,
                layers, target_scale_idx, random_intro_matrix,
            )
            self._plot_layer_effects(
                layer_results, baseline, target_scale,
                contrastive_prompts, random_baseline=random_baseline,
            )

        heatmap_result = {
            "layers": layers,
            "scales": scales,
            "intro_matrix": intro_matrix,
            "control_matrix": control_mean_matrix,            # legacy alias
            "control_mean_matrix": control_mean_matrix,
            "control_std_matrix": control_std_matrix,
            "z_matrix": z_matrix,
            "baseline": baseline,
            "baseline_intro": baseline["intro_diff"],         # legacy alias
            "baseline_control": baseline["control_mean"],     # legacy alias
            "random_intro_matrix": random_intro_matrix,
        }

        self._save_results_json(
            "heatmap", heatmap_result, baseline, contrastive_prompts,
            extra={"random_baseline": random_baseline,
                   "num_random_seeds": num_random_seeds if random_baseline else 0},
        )

        return heatmap_result
    def run_generation_experiment(
        self,
        layer_idx: int,
        magnitude: float = 1.0,
        token_pos: int = -1,
        contrastive_prompts: Tuple[str, str] = None,
        max_new_tokens: int = 50,
        steer_all_tokens: bool = False
    ):
        """Run generation experiment: sample actual text responses at temperature zero.

        Args:
            layer_idx: Layer to inject steering vector at
            magnitude: Scaling factor for steering vector
            token_pos: Token position to inject at (-1 for last)
            contrastive_prompts: Tuple of (prompt1, prompt2) for contrastive vector
            max_new_tokens: Maximum number of tokens to generate
            steer_all_tokens: If True, apply steering to all token positions

        Returns:
            Dictionary with all generated responses
        """
        print(f"\n{'='*70}")
        print(f"GENERATION EXPERIMENT (Temperature 0)")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        steer_mode = "all tokens" if steer_all_tokens else f"token pos {token_pos}"
        print(f"Layer: {layer_idx}, Scale: {magnitude}, Steer: {steer_mode}")
        print(f"Contrastive: '{contrastive_prompts[0][:30]}...' vs '{contrastive_prompts[1][:30]}...'")
        print(f"{'='*70}\n")

        # 1. Baseline - Introspection Question
        print(f"\n{'─'*70}")
        print("1. BASELINE - Introspection Question")
        print(f"{'─'*70}")
        print(f"Question: {self.introspection_question}")
        prompt_intro = self.format_prompt(self.introspection_question)
        response_baseline_intro = self.generate_response(prompt_intro, max_new_tokens)
        print(f"\nResponse: {response_baseline_intro}")

        # 2. With Steering - Introspection Question
        print(f"\n{'─'*70}")
        print(f"2. WITH STEERING (Layer {layer_idx}, Scale {magnitude}) - Introspection Question")
        print(f"{'─'*70}")
        print(f"Question: {self.introspection_question}")
        response_steering_intro = self.generate_response_with_steering(
            prompt_intro, layer_idx, magnitude, token_pos, contrastive_prompts, max_new_tokens, steer_all_tokens
        )
        print(f"\nResponse: {response_steering_intro}")

        # 3. Baseline - Control Question
        print(f"\n{'─'*70}")
        print("3. BASELINE - Control Question")
        print(f"{'─'*70}")
        print(f"Question: {self.control_question}")
        prompt_control = self.format_prompt(self.control_question)
        response_baseline_control = self.generate_response(prompt_control, max_new_tokens)
        print(f"\nResponse: {response_baseline_control}")

        # 4. With Steering - Control Question
        print(f"\n{'─'*70}")
        print(f"4. WITH STEERING (Layer {layer_idx}, Scale {magnitude}) - Control Question")
        print(f"{'─'*70}")
        print(f"Question: {self.control_question}")
        response_steering_control = self.generate_response_with_steering(
            prompt_control, layer_idx, magnitude, token_pos, contrastive_prompts, max_new_tokens, steer_all_tokens
        )
        print(f"\nResponse: {response_steering_control}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\nIntrospection Question: {self.introspection_question}")
        print(f"  Baseline:        {response_baseline_intro}")
        print(f"  With Steering:   {response_steering_intro}")
        print(f"\nControl Question: {self.control_question}")
        print(f"  Baseline:        {response_baseline_control}")
        print(f"  With Steering:   {response_steering_control}")
        print(f"\n{'='*70}\n")

        return {
            'introspection_baseline': response_baseline_intro,
            'introspection_steering': response_steering_intro,
            'control_baseline': response_baseline_control,
            'control_steering': response_steering_control
        }

    def _plot_layer_effects(
        self,
        layer_results: List[Dict],
        baseline: Dict,
        magnitude: float,
        contrastive_prompts: Tuple[str, str],
        random_baseline: bool = False,
    ):
        """Plot logit difference vs layer, with control battery band + z-score.

        Left axis: introspection logit-diff, control battery mean +/- std
        (shaded), and optional random-vector baseline band (feature #2).
        Right axis (twin): z-score of introspection against the control
        battery (feature #1). z ~= 0 means pure confusion; large |z| means
        introspection deviates specifically from the control distribution.

        Args:
            layer_results: List of per-layer result dicts
            baseline: Baseline stats dict (from run_baseline)
            magnitude: Magnitude used for steering
            contrastive_prompts: Tuple of (prompt1, prompt2) for plot title
            random_baseline: Whether to overlay the random-vector band
        """
        layers = [r['layer'] for r in layer_results]
        intro_diffs = [r['intro_diff'] for r in layer_results]
        control_means = [r['control_mean'] for r in layer_results]
        control_stds = [r['control_std'] for r in layer_results]
        z_scores = np.array([r['z_score'] for r in layer_results], dtype=float)
        baseline_intro = baseline["intro_diff"]
        baseline_control_mean = baseline["control_mean"]
        baseline_control_std = baseline["control_std"]

        fig, ax = plt.subplots(figsize=(15, 9))
        ax_z = ax.twinx()  # twin axis for z-score (feature #1)

        # Baseline references
        ax.axhline(y=baseline_intro, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline introspection: {baseline_intro:+.2f}')
        ax.axhline(y=baseline_control_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline control mean: {baseline_control_mean:+.2f}')

        # Control battery band: mean +/- std (feature #1)
        ctrl_means_arr = np.asarray(control_means)
        ctrl_stds_arr = np.asarray(control_stds)
        ax.fill_between(
            layers, ctrl_means_arr - ctrl_stds_arr, ctrl_means_arr + ctrl_stds_arr,
            color='red', alpha=0.18, label='Control battery band (+/- std)',
        )
        ax.plot(layers, control_means, 's-', linewidth=1.8, markersize=5, color='red',
                label='Control battery mean')

        # Optional random-vector baseline band (feature #2)
        if random_baseline and 'random_intro_diff' in layer_results[0]:
            rand_intro = [r['random_intro_diff'] for r in layer_results]
            rand_std = [r['random_intro_std'] for r in layer_results]
            rand_arr = np.asarray(rand_intro)
            rand_std_arr = np.asarray(rand_std)
            ax.fill_between(
                layers, rand_arr - rand_std_arr, rand_arr + rand_std_arr,
                color='gray', alpha=0.18, label='Random-vector baseline band',
            )
            ax.plot(layers, rand_intro, '^-', linewidth=1.5, markersize=5, color='gray',
                    label='Random-vector baseline')

        # Introspection
        ax.plot(layers, intro_diffs, 'o-', linewidth=2, markersize=5, color='blue',
                label='Introspection question')

        # Multi-trial error bars (intro)
        if len(layer_results[0]['all_intro_diffs']) > 1:
            intro_stds = [np.std(r['all_intro_diffs']) for r in layer_results]
            ax.errorbar(layers, intro_diffs, yerr=intro_stds, fmt='none', ecolor='blue',
                        capsize=4, alpha=0.4)

        # z-score on twin axis (feature #1)
        ax_z.plot(layers, z_scores, 'd--', linewidth=1.2, markersize=5, color='purple',
                  alpha=0.85, label='z-score (intro vs battery)')
        ax_z.axhline(y=0, color='purple', linestyle=':', linewidth=1, alpha=0.5)
        ax_z.set_ylabel('z-score (introspection vs control battery)', fontsize=11, color='purple')
        ax_z.tick_params(axis='y', labelcolor='purple')

        # Styling
        ax.set_xlabel('Layer Index', fontsize=13)
        ax.set_ylabel('Logit(Yes) - Logit(No)', fontsize=13)

        model_display = self.model_name.split("/")[-1]
        steering_info = f'Contrastive Steering: "{contrastive_prompts[0]}" vs "{contrastive_prompts[1]}" (strength={magnitude})'
        ax.set_title(f'LLM Introspection Experiment: {model_display}\n{steering_info}',
                     fontsize=13, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        y_min, y_max = ax.get_ylim()
        if y_max > 0:
            ax.axhspan(0, y_max, alpha=0.05, color='green')
        if y_min < 0:
            ax.axhspan(y_min, 0, alpha=0.05, color='orange')

        # Combined legend across both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_z.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')

        # Info text box
        n_controls = len(baseline.get("control_diffs", []) or self.control_questions)
        textstr = (
            f'Introspection: "{self.introspection_question}"\n\n'
            f'Control battery: {n_controls} questions (all "No"-expected)\n'
            f'First control: "{self.control_questions[0]}"'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()

        # Save plot
        suffix = "_battery"
        if random_baseline:
            suffix += "_random"
        filename = os.path.join(
            self.plot_dir, f"layer_sweep_scale{magnitude}{suffix}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n[Plot saved to: {filename}]")
        plt.show()

    def _plot_scale_effects(
        self,
        scale_results: List[Dict],
        baseline: Dict,
        layer_idx: int,
        contrastive_prompts: Tuple[str, str],
        random_baseline: bool = False,
    ):
        """Plot logit difference vs scale, with control band + z-score.

        Mirrors _plot_layer_effects but along the scale axis at one layer.

        Args:
            scale_results: List of per-scale result dicts
            baseline: Baseline stats dict (from run_baseline)
            layer_idx: Layer index where steering was applied
            contrastive_prompts: Tuple of (prompt1, prompt2) for the title
            random_baseline: Whether to overlay the random-vector band
        """
        scales = [r['scale'] for r in scale_results]
        intro_diffs = [r['intro_diff'] for r in scale_results]
        control_means = [r['control_mean'] for r in scale_results]
        control_stds = [r['control_std'] for r in scale_results]
        z_scores = np.array([r['z_score'] for r in scale_results], dtype=float)
        baseline_intro = baseline["intro_diff"]
        baseline_control_mean = baseline["control_mean"]
        baseline_control_std = baseline["control_std"]

        fig, ax = plt.subplots(figsize=(15, 9))
        ax_z = ax.twinx()  # twin axis for z-score (feature #1)

        # Baseline references
        ax.axhline(y=baseline_intro, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline introspection: {baseline_intro:+.2f}')
        ax.axhline(y=baseline_control_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Baseline control mean: {baseline_control_mean:+.2f}')

        # Control battery band: mean +/- std (feature #1)
        ctrl_means_arr = np.asarray(control_means)
        ctrl_stds_arr = np.asarray(control_stds)
        ax.fill_between(
            scales, ctrl_means_arr - ctrl_stds_arr, ctrl_means_arr + ctrl_stds_arr,
            color='red', alpha=0.18, label='Control battery band (+/- std)',
        )
        ax.plot(scales, control_means, 's-', linewidth=2, markersize=8, color='red',
                label='Control battery mean')

        # Optional random-vector baseline band (feature #2)
        if random_baseline and 'random_intro_diff' in scale_results[0]:
            rand_intro = [r['random_intro_diff'] for r in scale_results]
            rand_std = [r['random_intro_std'] for r in scale_results]
            rand_arr = np.asarray(rand_intro)
            rand_std_arr = np.asarray(rand_std)
            ax.fill_between(
                scales, rand_arr - rand_std_arr, rand_arr + rand_std_arr,
                color='gray', alpha=0.18, label='Random-vector baseline band',
            )
            ax.plot(scales, rand_intro, '^-', linewidth=1.8, markersize=6, color='gray',
                    label='Random-vector baseline')

        # Introspection
        ax.plot(scales, intro_diffs, 'o-', linewidth=2.5, markersize=8, color='blue',
                label='Introspection question')

        if len(scale_results[0]['all_intro_diffs']) > 1:
            intro_stds = [np.std(r['all_intro_diffs']) for r in scale_results]
            ax.errorbar(scales, intro_diffs, yerr=intro_stds, fmt='none', ecolor='blue',
                        capsize=5, alpha=0.4)

        # z-score on twin axis (feature #1)
        ax_z.plot(scales, z_scores, 'd--', linewidth=1.2, markersize=5, color='purple',
                  alpha=0.85, label='z-score (intro vs battery)')
        ax_z.axhline(y=0, color='purple', linestyle=':', linewidth=1, alpha=0.5)
        ax_z.set_ylabel('z-score (introspection vs control battery)', fontsize=11, color='purple')
        ax_z.tick_params(axis='y', labelcolor='purple')

        # Styling
        ax.set_xlabel('Steering Vector Scale', fontsize=13)
        ax.set_ylabel('Logit(Yes) - Logit(No)', fontsize=13)

        model_display = self.model_name.split("/")[-1]
        steering_info = f'Contrastive Steering: "{contrastive_prompts[0]}" - "{contrastive_prompts[1]}" at layer {layer_idx}'
        ax.set_title(f'Steering vector scale sweep\n{model_display}\n{steering_info}',
                     fontsize=13, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        y_min, y_max = ax.get_ylim()
        if y_max > 0:
            ax.axhspan(0, y_max, alpha=0.05, color='green')
        if y_min < 0:
            ax.axhspan(y_min, 0, alpha=0.05, color='orange')

        # Combined legend across both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_z.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')

        # Info text box
        n_controls = len(baseline.get("control_diffs", []) or self.control_questions)
        textstr = (
            f'Introspection: "{self.introspection_question}"\n\n'
            f'Control battery: {n_controls} questions (all "No"-expected)\n'
            f'First control: "{self.control_questions[0]}"'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

        plt.tight_layout()

        # Save plot
        suffix = "_battery"
        if random_baseline:
            suffix += "_random"
        filename = os.path.join(
            self.plot_dir, f"scale_sweep_layer{layer_idx}{suffix}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n[Plot saved to: {filename}]")
        plt.show()

    def _plot_heatmaps(
        self,
        intro_matrix: np.ndarray,
        control_mean_matrix: np.ndarray,
        control_std_matrix: np.ndarray,
        z_matrix: np.ndarray,
        layers: List[int],
        scales: List[float],
        baseline: Dict,
        contrastive_prompts: Tuple[str, str],
        random_intro_matrix: Optional[np.ndarray] = None,
    ):
        """Plot heatmaps for intro, control-battery mean, and z-score (feature #1).

        Optionally includes a 4th panel for the random-vector baseline's
        introspection matrix (feature #2). Layers run along x, scales along y.

        Args:
            intro_matrix: Introspection logit-diffs (layers x scales)
            control_mean_matrix: Control battery mean (layers x scales)
            control_std_matrix: Control battery std (layers x scales) -- shown
                as a text overlay of the max-std condition
            z_matrix: z-score of intro vs control battery (layers x scales)
            layers: Layer indices
            scales: Scale values
            baseline: Baseline stats dict (for annotations)
            contrastive_prompts: Tuple of (prompt1, prompt2) for title
            random_intro_matrix: Optional random-baseline intro matrix (layers x scales)
        """
        model_display = self.model_name.split("/")[-1]
        model_short = model_display.replace(".", "_")

        # Transpose: layers horizontal (x-axis), scales vertical (y-axis)
        intro_T = intro_matrix.T
        ctrl_T = control_mean_matrix.T
        z_T = np.ma.masked_invalid(z_matrix).T

        # Figure layout: 3 or 4 side-by-side panels
        n_panels = 3 if random_intro_matrix is None else 4
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 8))

        # Shared vmin/vmax for the intro / control panels (same units).
        vmin = min(intro_matrix.min(), control_mean_matrix.min())
        vmax = max(intro_matrix.max(), control_mean_matrix.max())
        norm = _diverging_norm(vmin, vmax)
        cmap = 'RdBu_r'  # Red for positive (Yes), Blue for negative (No)

        extent = [layers[0], layers[-1], scales[0], scales[-1]]

        # Panel 1: Introspection
        im1 = axes[0].imshow(intro_T, aspect='auto', cmap=cmap, norm=norm,
                             extent=extent, origin='lower')
        axes[0].set_xlabel('Layer Index', fontsize=12)
        axes[0].set_ylabel('Steering Vector Scale', fontsize=12)
        axes[0].set_title(f'Introspection\n"{self.introspection_question}"',
                          fontsize=11, fontweight='bold', pad=15)
        axes[0].text(0.02, 0.98, f'Baseline: {baseline["intro_diff"]:+.2f}',
                     transform=axes[0].transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 2: Control battery mean
        im2 = axes[1].imshow(ctrl_T, aspect='auto', cmap=cmap, norm=norm,
                             extent=extent, origin='lower')
        axes[1].set_xlabel('Layer Index', fontsize=12)
        axes[1].set_ylabel('Steering Vector Scale', fontsize=12)
        axes[1].set_title(f'Control battery mean ({len(self.control_questions)} questions)',
                          fontsize=11, fontweight='bold', pad=15)
        axes[1].text(0.02, 0.98,
                     f'Baseline mean: {baseline["control_mean"]:+.2f}\n'
                     f'Baseline std: {baseline["control_std"]:.2f}',
                     transform=axes[1].transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 3: z-score (feature #1). Symmetric diverging norm centered at 0
        # with a useful |z|~3 cap so saturated regions read clearly.
        z_finite = z_matrix[~np.isnan(z_matrix)]
        if z_finite.size == 0:
            zlim = 1.0
        else:
            zlim = min(6.0, max(3.0, np.nanmax(np.abs(z_finite))))
        z_norm = _diverging_norm(-zlim, zlim)
        im3 = axes[2].imshow(z_T, aspect='auto', cmap='Purples',
                             norm=z_norm, extent=extent, origin='lower')
        axes[2].set_xlabel('Layer Index', fontsize=12)
        axes[2].set_ylabel('Steering Vector Scale', fontsize=12)
        axes[2].set_title('z-score (intro vs control battery)\n'
                          'dark = introspection stands out',
                          fontsize=11, fontweight='bold', pad=15)

        # Panel 4 (optional): random-vector baseline intro (feature #2)
        if random_intro_matrix is not None:
            rand_T = random_intro_matrix.T
            r_vmin = min(intro_matrix.min(), random_intro_matrix.min())
            r_vmax = max(intro_matrix.max(), random_intro_matrix.max())
            r_norm = _diverging_norm(r_vmin, r_vmax)
            axes[3].imshow(rand_T, aspect='auto', cmap=cmap, norm=r_norm,
                           extent=extent, origin='lower')
            axes[3].set_xlabel('Layer Index', fontsize=12)
            axes[3].set_ylabel('Steering Vector Scale', fontsize=12)
            axes[3].set_title('Random-vector baseline\n(matched norm)',
                              fontsize=11, fontweight='bold', pad=15)

        # Main title
        steering_info = f'Contrastive: "{contrastive_prompts[0]}" vs "{contrastive_prompts[1]}"'
        fig.suptitle(f'LLM Introspection Heatmap: {model_display}\n{steering_info}',
                     fontsize=13, fontweight='bold', y=0.98)

        # Shared colorbar for intro / control panels (logit-diff units)
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.55, 0.012, 0.32])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_label('Logit(Yes) - Logit(No)', fontsize=11)

        # Separate colorbar for the z-score panel
        cbar_z_ax = fig.add_axes([0.90, 0.10, 0.012, 0.32])
        cbar_z = fig.colorbar(im3, cax=cbar_z_ax)
        cbar_z.set_label('z-score', fontsize=11)

        plt.tight_layout(rect=[0, 0, 0.88, 1])

        # Save plot
        suffix = "_battery"
        if random_intro_matrix is not None:
            suffix += "_random"
        filename = os.path.join(self.plot_dir, f"heatmap{suffix}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n[Heatmap saved to: {filename}]")
        plt.show()


def list_models():
    """Print available models organized by family."""
    print("\n=== Available Models ===\n")

    # Group by family
    families = {}
    for shortcut, config in MODEL_CONFIGS.items():
        family = config.get("family", "Other")
        if family not in families:
            families[family] = []
        families[family].append((shortcut, config))

    # Print Qwen family
    if "Qwen2.5" in families:
        print("Qwen2.5-Instruct Family (6 sizes):")
        for shortcut, config in sorted(families["Qwen2.5"], key=lambda x: x[1]["params"]):
            print(f"  {shortcut:20s} : {config['name']:50s} ({config['params']})")
        print()

    # Print Llama family
    if "Llama-3.x" in families:
        print("Llama 3.x Family (3 sizes):")
        for shortcut, config in sorted(families["Llama-3.x"], key=lambda x: x[1]["params"]):
            print(f"  {shortcut:20s} : {config['name']:50s} ({config['params']})")
        print()

    # Print Mistral family
    if "Mistral" in families:
        print("Mistral Family (1 size):")
        for shortcut, config in sorted(families["Mistral"], key=lambda x: x[1]["params"]):
            print(f"  {shortcut:20s} : {config['name']:50s} ({config['params']})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="LLM Introspection Experiment - Test emergence of introspection across model sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model test
  python introspection.py --model qwen2.5-0.5b
  python introspection.py --model qwen2.5-7b --trials 3

  # Test with custom layers and magnitude
  python introspection.py --model qwen2.5-3b --layers 0 18 35 --scale 5

  # Generate heatmap sweeping over layers and scales
  python introspection.py --model qwen2.5-7b --heatmap
  python introspection.py --model qwen2.5-3b --heatmap --heatmap-scales 0 2 4 6 8 10

  # Generate actual text responses (temperature 0)
  python introspection.py --model qwen2.5-7b --generate --layer 14 --scale 8.0

  # List all available models
  python introspection.py --list-models
        """
    )
    parser.add_argument("--model", default="qwen2.5-0.5b",
                       help="Model name or shortcut (default: qwen2.5-0.5b)")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                       help="Layers to test (default: all layers)")
    parser.add_argument("--scale", type=float, default=1.0,
                       help="Scaling factor for steering vector (default: 1.0)")
    parser.add_argument("--trials", type=int, default=1,
                       help="Number of trials per condition (default: 1)")
    parser.add_argument("--token-pos", type=int, default=-1,
                       help="Token position to inject steering vector (-1 for last)")
    parser.add_argument("--contrastive", nargs=2, metavar=("PROMPT1", "PROMPT2"),
                       default=["Hi! How are you?", "HI! HOW ARE YOU?"],
                       help="Contrastive prompts to generate steering vector")
    parser.add_argument("--scale-sweep", action="store_true",
                       help="Run scale sweep experiment instead of layer sweep")
    parser.add_argument("--scales", nargs="+", type=float, default=None,
                       help="Scales to test in sweep")
    parser.add_argument("--sweep-layer", type=int, default=None,
                       help="Layer to use for scale sweep (default: middle layer)")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available models and exit")
    parser.add_argument("--generate", action="store_true",
                       help="Generate actual text responses at temperature 0")
    parser.add_argument("--layer", type=int, default=None,
                       help="Layer index for --generate mode (default: middle layer)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate in --generate mode (default: 50)")
    parser.add_argument("--heatmap", action="store_true",
                       help="Run heatmap sweep over layers and scales")
    parser.add_argument("--heatmap-layers", nargs="+", type=int, default=None,
                       help="Layers to include in heatmap (default: all layers)")
    parser.add_argument("--heatmap-scales", nargs="+", type=float, default=None,
                       help="Scales to include in heatmap (default: 0-10)")
    parser.add_argument("--no-line-plots", action="store_true",
                       help="Skip line-plots when running --heatmap (default: "
                            "generate both heatmap and line plots)")
    parser.add_argument("--line-plot-layer", type=int, default=None,
                       help="Layer for the scale-sweep line plot in --heatmap "
                            "mode (default: middle layer)")
    parser.add_argument("--line-plot-scale", type=float, default=None,
                       help="Scale for the layer-sweep line plot in --heatmap "
                            "mode (default: max non-zero scale)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (default: only show progress)")
    parser.add_argument("--steer-all-tokens", action="store_true",
                       help="Apply steering to all token positions")

    # ---- Feature #1: control battery -------------------------------------
    parser.add_argument("--control-questions-file", type=str, default=None,
                       help="Path to a file with one control question per line. "
                            "Overrides the built-in 15-question battery.")
    parser.add_argument("--num-controls", type=int, default=None,
                       help="Randomly subsample this many control questions "
                            "from the battery (default: use all). Useful to "
                            "trade off runtime on large sweeps.")
    parser.add_argument("--legacy-single-control", action="store_true",
                       help="Reproduce the original Godet experiment with only "
                            "the '1+1=3?' control question (z-scores will be "
                            "nan since std=0). Disables the battery.")

    # ---- Feature #2: random-vector baseline -------------------------------
    parser.add_argument("--random-baseline", action="store_true",
                       help="Also run N random norm-matched vectors per "
                            "condition to isolate 'magnitude' effects from "
                            "'direction' (semantic-content) effects.")
    parser.add_argument("--num-random-seeds", type=int, default=DEFAULT_NUM_RANDOM_SEEDS,
                       help=f"Number of random vectors averaged per condition "
                            f"(default: {DEFAULT_NUM_RANDOM_SEEDS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help=f"Base seed for the random-vector generator "
                            f"(default: {DEFAULT_SEED})")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Name for the output subfolder under plots/ "
                            "(default: <model_short>_battery-random). All "
                            "figures from this run are saved to plots/<run_name>/.")

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        list_models()
        return

    # Resolve model shortcut if used
    model_name = MODEL_CONFIGS.get(args.model, {}).get("name", args.model)

    print(f"Starting experiment with model: {model_name}\n")

    # Prepare contrastive prompts
    contrastive_prompts = tuple(args.contrastive)
    if args.verbose:
        print(f"Using contrastive prompts:")
        print(f"  Prompt 1: {repr(contrastive_prompts[0])}")
        print(f"  Prompt 2: {repr(contrastive_prompts[1])}\n")

    # Assemble the control battery (feature #1)
    if args.control_questions_file:
        with open(args.control_questions_file, "r") as f:
            control_questions = [line.strip() for line in f if line.strip()]
        if not control_questions:
            raise ValueError(f"No control questions found in {args.control_questions_file}")
        print(f"Loaded {len(control_questions)} control questions from {args.control_questions_file}")
    else:
        control_questions = list(DEFAULT_CONTROL_QUESTIONS)

    if args.legacy_single_control:
        control_questions = [control_questions[0]]
        print("Legacy single-control mode: using only '"
              + control_questions[0] + "' as control (z-scores will be nan).")
    elif args.num_controls is not None and args.num_controls < len(control_questions):
        # Deterministic subsample so re-runs are reproducible.
        rng = random.Random(args.seed)
        control_questions = rng.sample(control_questions, args.num_controls)
        print(f"Subsampled to {len(control_questions)} control questions.")

    experiment = IntrospectionExperiment(
        model_name=model_name,
        verbose=args.verbose,
        control_questions=control_questions,
        seed=args.seed,
        run_name=args.run_name,
    )

    # Choose experiment type
    if args.heatmap:
        print("Running heatmap sweep experiment\n")
        experiment.run_heatmap_sweep(
            layers=args.heatmap_layers,
            scales=args.heatmap_scales,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            steer_all_tokens=args.steer_all_tokens,
            random_baseline=args.random_baseline,
            num_random_seeds=args.num_random_seeds,
            line_plots=not args.no_line_plots,
            line_plot_layer=args.line_plot_layer,
            line_plot_scale=args.line_plot_scale,
        )
    elif args.generate:
        num_layers = len(experiment.layer_modules)
        layer = args.layer if args.layer is not None else num_layers // 2

        print(f"Running generation experiment at layer {layer}")
        print(f"Max tokens: {args.max_tokens}\n")

        experiment.run_generation_experiment(
            layer_idx=layer,
            magnitude=args.scale,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            max_new_tokens=args.max_tokens,
            steer_all_tokens=args.steer_all_tokens
        )
    elif args.scale_sweep:
        num_layers = len(experiment.layer_modules)
        sweep_layer = args.sweep_layer if args.sweep_layer is not None else num_layers // 2
        scales = args.scales if args.scales is not None else [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

        print(f"Running scale sweep at layer {sweep_layer}")
        print(f"Scales to test: {scales}\n")

        experiment.run_scale_sweep(
            layer_idx=sweep_layer,
            scales=scales,
            num_trials=args.trials,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            plot=True,
            steer_all_tokens=args.steer_all_tokens,
            random_baseline=args.random_baseline,
            num_random_seeds=args.num_random_seeds,
        )
    else:
        experiment.run_full_experiment(
            layers=args.layers,
            magnitude=args.scale,
            num_trials=args.trials,
            token_pos=args.token_pos,
            contrastive_prompts=contrastive_prompts,
            plot=True,
            steer_all_tokens=args.steer_all_tokens,
            random_baseline=args.random_baseline,
            num_random_seeds=args.num_random_seeds,
        )


if __name__ == "__main__":
    main()
