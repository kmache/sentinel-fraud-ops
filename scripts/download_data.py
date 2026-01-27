import kagglehub
import shutil
import os
from pathlib import Path
import pandas as pd
import gc

def download_dataset(
    dataset_slug: str = "sonalisna/ieeefrauddetection",
    output_dir: str = "data/raw",
    force: bool = False
):
    """
    Downloads data via KaggleHub. 
    If force=True, it wipes the specific Kaggle cache for this dataset to ensure a fresh download.
    Then merges tables and saves to 'data/raw'.
    """
    dest_path = Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    merged_file_path = dest_path / "train_raw.csv"
   
    # 1. Check Local Project File
    if merged_file_path.exists() and not force:
        print(f"✅ Data already exists at {merged_file_path}. Skipping.", flush=True)
        return

    print(f"⬇️  Initializing Download (via KaggleHub)...", flush=True)
    
    try:
        # 2. Get the path (Instant if already cached)
        cache_path = kagglehub.dataset_download(dataset_slug)
        
        # 3. FORCE RE-DOWNLOAD LOGIC
        if force:
            print(f"🧹 Force enabled: Cleaning cache at {cache_path}...", flush=True)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path) # Delete the cached folder
                print(f"   -> Cache deleted.", flush=True)
            
            print(f"⬇️  Re-downloading fresh copy from Kaggle...", flush=True)
            # This call will now trigger the real progress bar since cache is gone
            cache_path = kagglehub.dataset_download(dataset_slug)
            print(f"✅ Downloaded fresh to: {cache_path}", flush=True)
        else:
            print(f"✅ Using cached version at: {cache_path}", flush=True)

        # 4. Read and Merge
        print("🔄 Reading and Merging CSVs (This uses ~3GB RAM)...", flush=True)
        tnx_path = os.path.join(cache_path, "train_transaction.csv")
        id_path = os.path.join(cache_path, "train_identity.csv")

        if not os.path.exists(tnx_path) or not os.path.exists(id_path):
             raise FileNotFoundError("Could not find train_transaction or train_identity in downloaded data.")

        df_tnx = pd.read_csv(tnx_path)
        print(f"   -> Loaded Transactions: {df_tnx.shape}", flush=True)
        
        df_id = pd.read_csv(id_path)
        print(f"   -> Loaded Identity:     {df_id.shape}", flush=True)
        
        df = pd.merge(df_tnx, df_id, on="TransactionID", how="left")
        print(f"   -> Merged Shape:        {df.shape}", flush=True)

        print("🧹 Cleaning up RAM...", flush=True)
        del df_id
        del df_tnx
        gc.collect()
        
        # 5. Save
        print(f"💾 Saving merged file to {merged_file_path}...", flush=True)
        print(f"   (This takes 1-2 minutes. Please wait...)", flush=True)
        df.to_csv(merged_file_path, index=False)
        del df
        gc.collect()
        print(f"✅ Saved successfully.", flush=True)


        # 6. Copy Source Files
        print(f"📦 Copying original source files to {dest_path}...", flush=True)
        
        source_files_to_keep = ["train_transaction.csv", "train_identity.csv"]
        
        for file_name in os.listdir(cache_path):
            source = os.path.join(cache_path, file_name)
            destination = dest_path / file_name
            
            if file_name in source_files_to_keep:
                shutil.copy2(source, destination)
                print(f"   -> Copied: {file_name}", flush=True)
            else:
                print(f"   -> Skipped (Unused): {file_name}", flush=True)

        print("🎉 Success! Data is ready for Sentinel.", flush=True)
            
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        print("   (Make sure you have authorized your Kaggle token)", flush=True)

if __name__ == "__main__":
    # Change force=True to test the redownload logic
    download_dataset(output_dir="data/raw", force=True)