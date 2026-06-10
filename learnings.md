### Differecn between normal creating and storing vectors vs te inbulit ones.

The only difference is that real vector databases do this over millions of vectors using optimized nearest-neighbor algorithms instead of looping through a Python list.

### Why can't we simply use a very large chunk size?

Because unrelated information becomes part of the same chunk, reducing retrieval precision and increasing irrelevant context sent to the LLM.

### Production RAG Reality

### Many people think:

chunk_size=1000
chunk_overlap=200

is a magic number.

It isn't.

Chunk size depends on:

Legal Documents

Large chunks.

1000-1500

because context matters.

FAQs

Small chunks.

200-500

because answers are short.

Source Code

Often function-level chunking.

Not character chunking.

Enterprise Policies

Usually:

500-1000

with

100-200 overlap

Q1: Why do we need chunking if we already have embeddings?

Your answer:

we need to have find the closest answer

Better answer:

Embeddings work best when they represent a focused piece of information. If we create a single embedding for an entire document, the meaning of specific sections gets diluted. Chunking breaks a document into smaller meaningful sections so that retrieval can find the most relevant part instead of the whole document.

Interview version:

We use chunking because embeddings of large documents lose specificity. Chunking allows us to create embeddings for smaller, focused pieces of content, improving retrieval accuracy in RAG systems.

Q2: What happens if chunk size is too small?

Your answer:

information is distorted and hence cannot get the exact answer.

Good intuition.

Interview version:

If chunk size is too small, context gets fragmented. Important information may be split across multiple chunks, making retrieval less effective and causing the LLM to receive incomplete context.

Example:

Chunk 1:
Employees are entitled to 25

Chunk 2:
vacation days annually

The meaning is broken.

Q3: What happens if chunk size is too large?

Your answer:

information gets mixed up.

Correct.

Interview version:

If chunk size is too large, multiple topics can be combined into a single chunk. This reduces retrieval precision because unrelated information may be returned along with the relevant content.

Example:

Employee leave policy

Azure cloud services

SQL Server indexing

All inside one chunk.

Now retrieval becomes noisy.

Q4: Why do we use overlap?

Your answer:

to retain the context

Good.

Interview version:

Overlap preserves context across chunk boundaries. It ensures that information spanning multiple chunks is not lost and helps maintain semantic continuity during retrieval.

Example:

Without overlap:

Chunk 1:
Employees are entitled to 25

Chunk 2:
vacation days annually

With overlap:

Chunk 1:
Employees are entitled to 25

Chunk 2:
entitled to 25 vacation days annually

Much better.

Q5: Why is RecursiveCharacterTextSplitter preferred over fixed chunking?

This is the one you missed.

Think about these two approaches.

Fixed Chunking
text[0:100]
text[100:200]
text[200:300]

It doesn't care about:

paragraphs
sentences
words

It just cuts.

Example:

Employees are entitled to 25 vaca
tion days annually.

Meaning is destroyed.

RecursiveCharacterTextSplitter

It tries to preserve meaning.

It attempts splitting in this order:

Paragraph
↓
Sentence
↓
Word
↓
Character

Only when needed.

Therefore it keeps semantic boundaries intact.

Interview answer:

RecursiveCharacterTextSplitter is preferred because it tries to preserve the natural structure of the text by splitting on paragraphs, sentences, and words before resorting to character-level splitting. This results in more meaningful chunks and better retrieval quality compared to fixed chunking.
