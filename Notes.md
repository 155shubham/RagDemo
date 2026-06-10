✅ 1. Embeddings measure similarity, not correctness

They rank text that looks similar, not what is actually the right answer.


✅ 2. Exact keywords dominate ranking

"loyalty" in a wrong chunk > indirect meaning like "companion" in the correct chunk.


✅ 3. Indirect / implicit answers get missed easily

If your chunk doesn’t explicitly contain key terms, it will rank lower.


✅ 4. Top‑K is a recall bottleneck

Correct chunk may exist (like your #12) but gets dropped if K is small.


✅ 5. Multi-query only works if implemented correctly

You must retrieve per query and merge results, not pass a list as input.


✅ 6. Even after retrieval, ranking can still be wrong

Your latest example proved:

✅ chunk present
❌ but ranked lower




✅ 7. Strong but generic chunks fool the retriever

“Loyalty is a trait…” beats more specific but weaker answers.


✅ 8. Retriever ≠ reasoner

It cannot infer:
best companion → loyal




✅ 9. Reranker is the real game changer

Retriever finds candidates
✅ Reranker picks the correct one


✅ 10. Real fix = combine techniques
Best pipeline:
Multi-query → Top-K (20–50) → Reranker → LLM


🧠 One-line summary

RAG fails not because data is missing, but because the retriever can't recognize the right answer — reranking fixes that.