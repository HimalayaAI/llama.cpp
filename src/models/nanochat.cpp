#include "models.h"

static bool nanochat_has_value_embd(int il, int n_layer) {
    return il % 2 == (n_layer - 1) % 2;
}

static ggml_tensor * nanochat_layer_scalar(ggml_context * ctx0, ggml_tensor * t, int il) {
    return ggml_view_1d(ctx0, t, 1, il * ggml_element_size(t));
}

void llama_model_nanochat::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 15: type = LLM_TYPE_0_5B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_nanochat::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, 0);

    nanochat_resid_lambdas  = create_tensor(tn(LLM_TENSOR_NANOCHAT_RESID_LAMBDAS,  "weight"), {n_layer}, 0);
    nanochat_x0_lambdas     = create_tensor(tn(LLM_TENSOR_NANOCHAT_X0_LAMBDAS,     "weight"), {n_layer}, 0);
    nanochat_smear_gate     = create_tensor(tn(LLM_TENSOR_NANOCHAT_SMEAR_GATE,     "weight"), {24, 1}, 0);
    nanochat_smear_lambda   = create_tensor(tn(LLM_TENSOR_NANOCHAT_SMEAR_LAMBDA,   "weight"), {1}, 0);
    nanochat_backout_lambda = create_tensor(tn(LLM_TENSOR_NANOCHAT_BACKOUT_LAMBDA, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);

        const int flags = nanochat_has_value_embd(i, n_layer) ? 0 : TENSOR_NOT_REQUIRED;
        layer.nanochat_value_embd = create_tensor(tn(LLM_TENSOR_NANOCHAT_VALUE_EMBD, "weight", i), {n_embd, n_vocab}, flags);
        layer.nanochat_ve_gate    = create_tensor(tn(LLM_TENSOR_NANOCHAT_VE_GATE,    "weight", i), {12, n_head_kv}, flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_nanochat::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_nanochat::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    cb(inpL, "inp_norm", -1);

    if (n_tokens > 1) {
        ggml_tensor * first = ggml_view_2d(ctx0, inpL, n_embd, 1, inpL->nb[1], 0);
        ggml_tensor * tail  = ggml_view_2d(ctx0, inpL, n_embd, n_tokens - 1, inpL->nb[1], inpL->nb[1]);
        ggml_tensor * prev  = ggml_view_2d(ctx0, inpL, n_embd, n_tokens - 1, inpL->nb[1], 0);

        ggml_tensor * gate_inp = ggml_view_2d(ctx0, inpL, 24, n_tokens - 1, inpL->nb[1], inpL->nb[1]);
        ggml_tensor * gate     = ggml_mul_mat(ctx0, model.nanochat_smear_gate, gate_inp);
        gate = ggml_sigmoid(ctx0, gate);
        gate = ggml_mul(ctx0, gate, model.nanochat_smear_lambda);
        cb(gate, "inp_smear_gate", -1);

        tail = ggml_add(ctx0, tail, ggml_mul(ctx0, prev, gate));
        inpL = ggml_concat(ctx0, first, tail, 1);
        cb(inpL, "inp_smear", -1);
    }

    ggml_tensor * x0 = inpL;

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * x_backout = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * resid_lambda = nanochat_layer_scalar(ctx0, model.nanochat_resid_lambdas, il);
        ggml_tensor * x0_lambda    = nanochat_layer_scalar(ctx0, model.nanochat_x0_lambdas, il);

        inpL = ggml_add(ctx0,
                ggml_mul(ctx0, inpL, resid_lambda),
                ggml_mul(ctx0,   x0, x0_lambda));
        cb(inpL, "layer_resid_mix", il);

        ggml_tensor * inpSA = inpL;

        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            if (model.layers[il].nanochat_value_embd) {
                ggml_tensor * ve = ggml_get_rows(ctx0, model.layers[il].nanochat_value_embd, res->get_inp_tokens());
                ve = ggml_reshape_3d(ctx0, ve, n_embd_head, n_head_kv, n_tokens);
                cb(ve, "value_embd", il);

                ggml_tensor * gate_inp = ggml_view_2d(ctx0, cur, 12, n_tokens, cur->nb[1], 0);
                ggml_tensor * gate     = ggml_mul_mat(ctx0, model.layers[il].nanochat_ve_gate, gate_inp);
                gate = ggml_scale(ctx0, ggml_sigmoid(ctx0, gate), 3.0f);
                gate = ggml_reshape_3d(ctx0, gate, 1, n_head_kv, n_tokens);
                gate = ggml_repeat(ctx0, gate, Vcur);
                cb(gate, "value_embd_gate", il);

                Vcur = ggml_add(ctx0, Vcur, ggml_mul(ctx0, gate, ve));
                cb(Vcur, "Vcur_value_embd", il);
            }

            // Nanochat trains with R(-pos*theta), opposite of standard RoPE's R(+pos*theta).
            // Negate freq_scale so ggml computes cos/sin of -theta (cos unchanged, sin negated).
            const float nanochat_freq_scale = -freq_scale;

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, nanochat_freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, nanochat_freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Qcur = ggml_scale(ctx0, ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps), 1.2f);
            Kcur = ggml_scale(ctx0, ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps), 1.2f);

            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                      NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        if (il == n_layer / 2) {
            x_backout = cur;
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    if (x_backout != nullptr) {
        cur = ggml_sub(ctx0, cur, ggml_mul(ctx0, x_backout, model.nanochat_backout_lambda));
        cb(cur, "backout", -1);
    }

    cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
    cb(cur, "result_norm", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cur = ggml_scale(ctx0, cur, 1.0f / 15.0f);
    cur = ggml_tanh(ctx0, cur);
    cur = ggml_scale(ctx0, cur, 15.0f);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
