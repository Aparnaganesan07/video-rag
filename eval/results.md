Evaluation: 50 questions over 4 videos (104 chunks). A hit = right video and ground-truth time inside the chunk ±20s. Search spans all videos.

| Setup | Hit@1 | Hit@3 | Hit@5 | MRR | Latency/query (CPU) |
|---|---|---|---|---|---|
| transcript-only | 0.40 | 0.80 | 0.88 | 0.610 | 0.04s |
| + slides | 0.48 | 0.84 | 0.92 | 0.670 | 0.04s |
| + hybrid (BM25) | 0.54 | 0.90 | 1.00 | 0.733 | 0.04s |
| + reranker | 0.66 | 0.88 | 0.98 | 0.790 | 10.63s |

Hit@1 by question type:

| Setup | keyword (n=15) | paraphrase (n=26) | slide (n=9) |
|---|---|---|---|
| transcript-only | 0.53 | 0.35 | 0.33 |
| + slides | 0.53 | 0.42 | 0.56 |
| + hybrid (BM25) | 0.53 | 0.42 | 0.89 |
| + reranker | 0.73 | 0.54 | 0.89 |

End-to-end answers (openai/gpt-oss-120b, reasoning effort low, final setup):

- At least one citation within ±20s of the ground truth: **92%**
- First citation within ±20s: 90%
- Answered "not found": 2%
- Ungrounded (hallucinated) citations: 0
