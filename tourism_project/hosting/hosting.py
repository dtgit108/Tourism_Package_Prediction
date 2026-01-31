from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing Streamlit app files
    repo_id="Dtapkir/TourismPackagePrediction",  # Hugging Face Space repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                        # upload to root of the space
)
