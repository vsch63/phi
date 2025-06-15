from huggingface_hub import snapshot_download

# Set target folder for download
download_path = snapshot_download(
    repo_id="microsoft/phi-2",
    local_dir="phi2_model",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {download_path}")
