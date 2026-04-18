import os
import urllib.request
import yaml
import ssl

# Bypass SSL verification to avoid CERTIFICATE_VERIFY_FAILED error
ssl._create_default_https_context = ssl._create_unverified_context

def download_subset(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Could not read config.yaml: {e}")
        return
        
    raw_path = config['data']['raw_path']
    classes = config['data']['subset_classes']
    
    # Base URL for THUMOS14 UCF101 mirror
    base_url = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
    
    print("Downloading UCF101 subset (this might take a few minutes if first time)...")
    
    # Download up to 19 groups per class for a larger dataset
    for cls in classes:
        cls_dir = os.path.join(raw_path, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for group in range(1, 20):
            group_str = f"g{group:02d}"
            video_name = f"v_{cls}_{group_str}_c01.avi"
            url = base_url + video_name
            save_path = os.path.join(cls_dir, video_name)
            
            if not os.path.exists(save_path):
                try:
                    print(f"Downloading {video_name}...")
                    urllib.request.urlretrieve(url, save_path)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
            else:
                print(f"{video_name} already exists.")
    
if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    download_subset(config_file)
