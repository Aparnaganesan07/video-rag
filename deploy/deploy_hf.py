"""Upload the app to a Hugging Face Space (Docker SDK).

One-time setup (you do this, it needs your own token):
    1. Create a free account at https://huggingface.co
    2. Create a token with WRITE access at https://huggingface.co/settings/tokens
    3. In a terminal:  .\.venv\Scripts\hf.exe auth login      (paste the token when asked)

Then:
    python -m deploy.deploy_hf YOUR_HF_USERNAME

After the first upload, add your Groq key as a secret in the Space:
    Space page -> Settings -> Variables and secrets -> New secret: GROQ_API_KEY

Why not `git push` to the Space? The Space only needs the code + demo_data (not data/, .venv
or the demo GIF), and Hugging Face rejects plain-git binary files. upload_folder with an
allow-list sends exactly what the Docker build needs.
"""

import argparse

from huggingface_hub import HfApi

from app.config import ROOT_DIR

SPACE_NAME = "video-rag"

# Only what the Dockerfile needs. Everything else (data/, .venv, .env, media) stays local.
ALLOW = [
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "app/**/*.py",
    "ui/**/*.py",
    "eval/**",
    "tests/**/*.py",
    "pytest.ini",
    "demo_data/**/*.json",
    ".streamlit/config.toml",
]
IGNORE = ["**/__pycache__/**", "*.pyc", ".env"]


def main():
    parser = argparse.ArgumentParser(description="Deploy to a Hugging Face Space.")
    parser.add_argument("username", help="your Hugging Face username")
    args = parser.parse_args()
    repo_id = f"{args.username}/{SPACE_NAME}"

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT_DIR),
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        commit_message="Deploy from local",
    )
    # The Space's README carries the config block (sdk: docker, port) Hugging Face needs.
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=str(ROOT_DIR / "deploy" / "space_README.md"),
        path_in_repo="README.md",
        commit_message="Space README",
    )
    print(f"Uploaded. Build logs + app: https://huggingface.co/spaces/{repo_id}")
    print("Remember: add GROQ_API_KEY under Settings -> Variables and secrets (first deploy only).")


if __name__ == "__main__":
    main()
