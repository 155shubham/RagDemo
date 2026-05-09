# Create project

mkdir ragdemo
cd ragdemo

# Initialize project

uv init

# Create virtual environment

uv venv

# Activate venv (Linux/macOS)

source .venv/bin/activate

# Activate venv (Windows PowerShell)

# .venv\Scripts\Activate.ps1

# Add dependencies

uv add langchain
uv add langchain-community
uv add langchain-openai
uv add faiss-cpu
uv add tiktoken
uv add python-dotenv

# OR add all together

uv add langchain langchain-community langchain-openai faiss-cpu tiktoken python-dotenv
