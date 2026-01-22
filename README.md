# Test Time Training Models

Heavily WIP. May never be finished. Oh well!

An (attempted!) implementation of "test-time training" language models for long context in PyTorch, where an inner memory module learns to encode memory over the sequence by optimizing its weights as it processes tokens, introduced by [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) by Sun, et. al.

This likely won't work because I am not good at computer, but it's worth a shot anyway.

This framework will use the Hugging Face ecosystem, including the Transformers `Trainer`. Easier this way.


---

# Installation

The usual.

`pip3 install -r requirements.txt`

Two libraries in the requirements file are custom kernels from others:

- [**Accelerated Scan**](https://github.com/proger/accelerated-scan)
- [**Causal Conv1d**](https://github.com/Dao-AILab/causal-conv1d)

Please make sure your GPUs can support these libraries. I'll add alternative native versions later.

---

# Architecture Implementation Checklist

- [X] [**Learning to (Learn at Test Time): RNNs with Expressive Hidden States**](https://arxiv.org/abs/2407.04620) (copied from the [official implementation](https://github.com/test-time-training/ttt-lm-pytorch))
- [X] [**Titans: Learning to Memorize at Test Time**](https://arxiv.org/abs/2501.00663) (LMM variant only at the moment)
- [ ] [**Test-Time Training Done Right**](https://arxiv.org/abs/2505.23884)
- [ ] [**ATLAS: Learning to Optimally Memorize the Context at Test Time**](https://arxiv.org/abs/2505.23735)
- [ ] [**TNT: Improving Chunkwise Training For Test-Time Memorization**](https://arxiv.org/abs/2511.07343)
- [ ] [**ViT³: Unlocking Test-Time Training in Vision**](https://arxiv.org/abs/2512.01643)
- [ ] [**It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization**](https://arxiv.org/abs/2504.13173) (all variants)
- [ ] [**Nested Learning: The Illusion of Deep Learning Architecture**](https://arxiv.org/abs/2512.24695)

Will add more papers as I find them.