"""finetune — SFT prep, train configs, export (Skill C, Phase 4–5).

    prep.py    build the SFT dataset from consensus labels + corruption
               augmentation + hard cases + explicit abstention examples. REUSES
               tasks/ split logic so train can never touch test stations/times.
    configs    LoRA/QLoRA run configs (r, α, lr, epochs, target modules).
    export     merge adapter -> GGUF (Q8 and Q4) for local llama.cpp eval.

Training itself runs in the cloud (Colab T4 / rented 4090); nothing here pulls
torch onto this machine. The curriculum is three runs: (A) mechanics/overfit,
(B) the real stratified run, (C) exactly one ablation. See §4 of the plan.

Guardrail #5 — prediction creep: the moment a training example asks the model to
know something not derivable from the input text, delete it.
"""
