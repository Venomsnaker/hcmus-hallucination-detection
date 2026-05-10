# ---------------------------------------------------------------------------
# Helpers shared by the VLM-specific (Qwen3-VL) gradient/embedding adaptations
# ---------------------------------------------------------------------------

def _get_image_token_id(model, processor):
    """Return the placeholder token id used by the VLM for image patch slots."""
    cfg = getattr(model, 'config', None)
    if cfg is not None:
        for attr in ('image_token_id', 'image_token_index'):
            tid = getattr(cfg, attr, None)
            if tid is not None:
                return int(tid)
    tok = getattr(processor, 'tokenizer', None)
    if tok is not None:
        for piece in ('<|image_pad|>', '<|vision_pad|>', '<image>'):
            try:
                tid = tok.convert_tokens_to_ids(piece)
            except Exception:
                tid = None
            if tid is not None and tid != getattr(tok, 'unk_token_id', -1):
                return int(tid)
    raise RuntimeError('Could not determine the image_token_id for this VLM.')


def _locate_language_model(model):
    """Locate the inner causal-LM submodule of a Qwen3-VL style model."""
    for parent in (model, getattr(model, 'model', None)):
        if parent is None:
            continue
        lm = getattr(parent, 'language_model', None)
        if lm is not None:
            return lm
    raise RuntimeError('Could not locate the language_model submodule on the VLM.')


def _forward_with_input_embeds_override(model, model_inputs, override_embeds=None):
    """
    Run a single VLM forward, intercepting the merged ``inputs_embeds`` that the
    multimodal model passes to its inner language model.

    * If ``override_embeds`` is None, the merged embeddings are detached and turned
      into a leaf tensor (``requires_grad=True``) so gradients can be taken w.r.t.
      them. The leaf is returned alongside the model output.
    * If ``override_embeds`` is provided, those embeddings replace the merged
      embeddings before the LLM forward (used for IG path interpolation).
    """
    captured = {}

    def pre_hook(module, args, kwargs):
        new_args = list(args)
        new_kwargs = dict(kwargs)

        embeds = new_kwargs.get('inputs_embeds', None)
        embeds_in_kwargs = embeds is not None
        if not embeds_in_kwargs and len(new_args) > 0 and isinstance(new_args[0], torch.Tensor):
            embeds = new_args[0]

        if embeds is None:
            return tuple(new_args), new_kwargs

        if override_embeds is not None:
            replacement = override_embeds
        else:
            replacement = embeds.detach().requires_grad_(True)
        captured['inputs_embeds'] = replacement

        if embeds_in_kwargs:
            new_kwargs['inputs_embeds'] = replacement
        else:
            new_args[0] = replacement
        return tuple(new_args), new_kwargs

    lm = _locate_language_model(model)
    handle = lm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        output = model(**model_inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()

    return output, captured.get('inputs_embeds', None)


def _prepare_context_inputs(context_messages, answer_messages, model_bundle):
    """Tokenize/processor-encode and move tensors to device. Returns dict + lengths."""
    processor, device = model_bundle.processor, model_bundle.device

    c_ids = processor.apply_chat_template(
        context_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt',
    )
    a_ids = processor.apply_chat_template(
        answer_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt',
    )
    c_ids = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c_ids.items()}

    c_length = c_ids['input_ids'].shape[1]
    a_length = a_ids['input_ids'].shape[1]
    return c_ids, c_length, a_length


def _answer_logprob_scalar(logits, c_length, a_length, target_ids):
    """
    Sum log P(y_t | y_{<t}, context) over the assistant/answer span, matching the
    indexing convention of ``extract_features_qwen3``.
    """
    ans_logits = logits[c_length - a_length: c_length - 1, :].float()
    log_probs = torch.log_softmax(ans_logits, dim=-1)
    target_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return target_log_probs.sum(), target_log_probs


# ---------------------------------------------------------------------------
# Adaptation 1 — Modality Reliance Ratio (the "blindness" ratio)
# ---------------------------------------------------------------------------

def extract_features_qwen3_modality_ratio(
    context_messages: list,
    answer_messages: list,
    model_bundle: Qwen3ModelBundle):
    """
    Visual-vs-textual gradient sensitivity for each input position.

    For the answer span y, we differentiate the scalar
        S = sum_t log P(y_t | y_{<t}, v, t_context)
    with respect to the merged input embeddings E in R^{L x d}. Splitting E by
    the image-token mask yields:
        ||grad_v S||_2  (visual sensitivity)
        ||grad_t S||_2  (textual sensitivity)
    The "modality reliance ratio" is visual / textual; values << 1 flag the
    model as "blind" to the image at this generation step.
    """
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    c_ids, c_length, a_length = _prepare_context_inputs(context_messages, answer_messages, model_bundle)
    if a_length < 2:
        raise ValueError('Answer must contain at least 2 tokens for next-token prediction.')

    image_token_id = _get_image_token_id(model, processor)
    input_ids_flat = c_ids['input_ids'].squeeze(0)
    image_mask = (input_ids_flat == image_token_id)
    text_mask = ~image_mask
    target_ids = input_ids_flat[c_length - a_length + 1:]

    model.eval()
    with torch.set_grad_enabled(True):
        output, inputs_embeds = _forward_with_input_embeds_override(model, c_ids)
        if inputs_embeds is None:
            raise RuntimeError('Failed to capture merged inputs_embeds from the VLM.')

        logits = output.logits.squeeze(0)
        scalar, target_log_probs = _answer_logprob_scalar(logits, c_length, a_length, target_ids)

        grad = torch.autograd.grad(scalar, inputs_embeds, retain_graph=False)[0]
        grad = grad.squeeze(0)  # [seq, hidden]

        per_pos_norm = grad.norm(dim=-1)               # [seq]
        visual_grad_norm = per_pos_norm[image_mask]    # [num_visual]
        text_grad_norm = per_pos_norm[text_mask]       # [num_text]

        visual_total = visual_grad_norm.pow(2).sum().sqrt()
        text_total = text_grad_norm.pow(2).sum().sqrt()
        modality_ratio = visual_total / (text_total + 1e-12)

    out = {
        'visual_grad_norm': visual_grad_norm.detach().to('cpu'),
        'text_grad_norm': text_grad_norm.detach().to('cpu'),
        'visual_total': visual_total.detach().to('cpu'),
        'text_total': text_total.detach().to('cpu'),
        'modality_ratio': modality_ratio.detach().to('cpu'),
        'answer_logprob': target_log_probs.detach().to('cpu'),
        'image_mask': image_mask.detach().to('cpu'),
    }

    del c_ids, output, inputs_embeds, grad, logits, scalar, target_log_probs
    return out


# ---------------------------------------------------------------------------
# Adaptation 2 — Integrated Gradients for cross-modal attribution
# ---------------------------------------------------------------------------

def extract_features_qwen3_integrated_gradients(
    context_messages: list,
    answer_messages: list,
    model_bundle: Qwen3ModelBundle,
    n_steps: int = 20,
    baseline: str = 'black_image'):
    """
    Riemann-approximated Integrated Gradients along the straight-line path from a
    visual baseline (black image / zero pixels) to the actual visual input,
    while keeping textual embeddings fixed.

    For each interpolation alpha in {1/N, ..., 1}:
        E(alpha) = E_base + alpha * (E_actual - E_base)   (only image-token rows
                                                           differ; text rows are
                                                           identical between the
                                                           two captures)
        g(alpha) = d/dE  sum_t log P(y_t | E(alpha))

    IG(E) = (E_actual - E_base) * mean_alpha g(alpha)

    Returns IG attribution per input position. Visual rows of the attribution
    map answer "which image patches drove this generation"; their summed mass
    relative to the text rows answers "is the model attributing to the image,
    or to the conversation history?".
    """
    if baseline not in ('black_image', 'zero_pixels'):
        raise ValueError(f'Unsupported baseline: {baseline}')

    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    c_ids, c_length, a_length = _prepare_context_inputs(context_messages, answer_messages, model_bundle)
    if 'pixel_values' not in c_ids or c_ids['pixel_values'] is None:
        raise ValueError('Integrated Gradients requires pixel_values in the inputs.')
    if a_length < 2:
        raise ValueError('Answer must contain at least 2 tokens for next-token prediction.')

    image_token_id = _get_image_token_id(model, processor)
    input_ids_flat = c_ids['input_ids'].squeeze(0)
    image_mask = (input_ids_flat == image_token_id)
    target_ids = input_ids_flat[c_length - a_length + 1:]

    model.eval()

    # 1) Capture the merged embeddings for the actual image and for the baseline.
    with torch.no_grad():
        _, embeds_actual = _forward_with_input_embeds_override(model, c_ids)

        baseline_inputs = dict(c_ids)
        baseline_inputs['pixel_values'] = torch.zeros_like(c_ids['pixel_values'])
        _, embeds_baseline = _forward_with_input_embeds_override(model, baseline_inputs)

    if embeds_actual is None or embeds_baseline is None:
        raise RuntimeError('Failed to capture merged inputs_embeds for IG.')

    embeds_actual = embeds_actual.detach()
    embeds_baseline = embeds_baseline.detach()
    diff = embeds_actual - embeds_baseline  # zero on text rows, non-zero on image rows

    # 2) Riemann right-rule path integral of gradients.
    grad_accum = torch.zeros_like(embeds_actual)
    alphas = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=embeds_actual.device)[1:]

    with torch.set_grad_enabled(True):
        for alpha in alphas:
            interp = (embeds_baseline + float(alpha) * diff).detach().requires_grad_(True)
            output, _ = _forward_with_input_embeds_override(model, c_ids, override_embeds=interp)
            logits = output.logits.squeeze(0)
            scalar, _ = _answer_logprob_scalar(logits, c_length, a_length, target_ids)
            g = torch.autograd.grad(scalar, interp, retain_graph=False)[0]
            grad_accum = grad_accum + g.detach()
            del output, logits, scalar, g, interp

    grad_avg = grad_accum / float(n_steps)
    ig_attribution = (diff * grad_avg).squeeze(0)        # [seq, hidden]

    visual_ig = ig_attribution[image_mask]               # [num_visual, hidden]
    text_ig = ig_attribution[~image_mask]                # [num_text,   hidden]

    visual_ig_per_token_signed = visual_ig.sum(dim=-1)   # [num_visual]
    visual_ig_per_token_abs = visual_ig.abs().sum(dim=-1)
    text_ig_per_token_abs = text_ig.abs().sum(dim=-1)

    visual_mass = visual_ig_per_token_abs.sum()
    text_mass = text_ig_per_token_abs.sum()
    visual_attribution_share = visual_mass / (visual_mass + text_mass + 1e-12)

    # If the processor exposed image_grid_thw, hand it back so the caller can
    # reshape ``visual_ig_per_token_*`` into the 2D patch grid for visualization.
    image_grid_thw = c_ids.get('image_grid_thw', None)
    if torch.is_tensor(image_grid_thw):
        image_grid_thw = image_grid_thw.detach().to('cpu')

    out = {
        'visual_ig_per_token_signed': visual_ig_per_token_signed.detach().to('cpu'),
        'visual_ig_per_token_abs': visual_ig_per_token_abs.detach().to('cpu'),
        'text_ig_per_token_abs': text_ig_per_token_abs.detach().to('cpu'),
        'visual_mass': visual_mass.detach().to('cpu'),
        'text_mass': text_mass.detach().to('cpu'),
        'visual_attribution_share': visual_attribution_share.detach().to('cpu'),
        'image_mask': image_mask.detach().to('cpu'),
        'image_grid_thw': image_grid_thw,
        'n_steps': n_steps,
    }

    del c_ids, embeds_actual, embeds_baseline, diff, grad_accum, grad_avg, ig_attribution, visual_ig, text_ig
    return out


# ---------------------------------------------------------------------------
# Adaptation 3 — Density-Aware Embedding Calibration
# ---------------------------------------------------------------------------

def extract_features_qwen3_density(
    context_messages: list,
    answer_messages: list,
    model_bundle: Qwen3ModelBundle,
    k: int = 5,
    reference: str = 'visual'):
    """
    Geometric density of each visual token's embedding in the merged input
    representation.

    For the visual rows V in R^{n x d} of the merged inputs_embeds (the rows at
    image-token positions, post visual encoder + projector), we compute:

        knn_mean_dist[i]  = mean of the k smallest L2 distances from V[i]
                            to the chosen reference set
        local_variance[i] = mean coordinate-wise variance of those k neighbours
        density_score[i]  = 1 / (knn_mean_dist[i] + eps)

    High density / low local variance => V[i] sits in a generic, oversampled
    region of the latent space (a likely fallback "safe" embedding); low
    density / high local variance => V[i] is a distinct, sharp visual feature.
    """
    if reference not in ('visual', 'all'):
        raise ValueError(f'Unsupported reference set: {reference}')

    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    c_ids, _, _ = _prepare_context_inputs(context_messages, answer_messages, model_bundle)

    image_token_id = _get_image_token_id(model, processor)
    input_ids_flat = c_ids['input_ids'].squeeze(0)
    image_mask = (input_ids_flat == image_token_id)

    model.eval()
    with torch.no_grad():
        _, inputs_embeds = _forward_with_input_embeds_override(model, c_ids)
    if inputs_embeds is None:
        raise RuntimeError('Failed to capture merged inputs_embeds for density analysis.')

    embeds = inputs_embeds.squeeze(0).detach().float()    # [seq, hidden]
    visual_embeds = embeds[image_mask]                    # [num_visual, hidden]
    n = visual_embeds.shape[0]

    if n == 0:
        empty = torch.zeros(0)
        return {
            'visual_embeds': visual_embeds.detach().to('cpu'),
            'knn_mean_dist': empty,
            'local_variance': empty,
            'density_score': empty,
            'image_mask': image_mask.detach().to('cpu'),
            'image_grid_thw': c_ids.get('image_grid_thw', None),
        }

    if reference == 'visual':
        ref = visual_embeds
    else:
        ref = embeds

    # Pairwise L2 distances from each visual token to the reference set.
    dists = torch.cdist(visual_embeds, ref)               # [n, m]

    # Mask out self-distance when reference == 'visual' (diagonal).
    if reference == 'visual':
        diag_idx = torch.arange(n, device=dists.device)
        dists[diag_idx, diag_idx] = float('inf')

    kk = min(k, max(dists.shape[1] - (1 if reference == 'visual' else 0), 1))
    topk_dists, topk_idx = dists.topk(kk, largest=False, dim=-1)   # [n, kk]
    knn_mean_dist = topk_dists.mean(dim=-1)                        # [n]

    neighbor_embs = ref[topk_idx]                                  # [n, kk, hidden]
    local_variance = neighbor_embs.var(dim=1, unbiased=False).mean(dim=-1)  # [n]

    density_score = 1.0 / (knn_mean_dist + 1e-12)

    image_grid_thw = c_ids.get('image_grid_thw', None)
    if torch.is_tensor(image_grid_thw):
        image_grid_thw = image_grid_thw.detach().to('cpu')

    out = {
        'visual_embeds': visual_embeds.detach().to('cpu'),
        'knn_mean_dist': knn_mean_dist.detach().to('cpu'),
        'local_variance': local_variance.detach().to('cpu'),
        'density_score': density_score.detach().to('cpu'),
        'image_mask': image_mask.detach().to('cpu'),
        'image_grid_thw': image_grid_thw,
        'k': kk,
    }

    del c_ids, inputs_embeds, embeds, visual_embeds, dists, topk_dists, topk_idx, neighbor_embs
    return out