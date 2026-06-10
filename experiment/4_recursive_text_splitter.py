from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Employees are entitled to 25 vacation days annually.
Employees are also entitled to 10 sick leaves annually.
Azure provides cloud services.
SQL Server stores relational data.
"""

splitter_medium = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
chunks_medium = splitter_medium.split_text(text=text)

splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks_large = splitter_large.split_text(text=text)

splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5
)
chunks_small = splitter_small.split_text(text=text)

print("\n--------medium spitter--------")
for i, chunk in enumerate(chunks_medium):
    print(f"\nChunk {i+1}")
    print(chunk)

print("\n---------large spitter-------")
for i, chunk in enumerate(chunks_large):
    print(f"\nChunk {i+1}")
    print(chunk)

print("\n--------small spitter-------")
for i, chunk in enumerate(chunks_small):
    print(f"\nChunk {i+1}")
    print(chunk)
