import json

with open('experiment/rag.ipynb', 'r') as f:
    nb = json.load(f)

# Insert chunks cell before embeddings (at position 4)
chunks_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'chunks-cell',
    'metadata': {},
    'outputs': [],
    'source': [
        '# Create chunks\n',
        'from langchain_text_splitters import RecursiveCharacterTextSplitter\n',
        'text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)\n',
        'chunks = text_splitter.split_documents(documents=documents)\n',
        'print(f"Created {len(chunks)} chunk(s)")'
    ]
}

nb['cells'].insert(4, chunks_cell)

with open('experiment/rag.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Fixed notebook paths and Document conversion")
