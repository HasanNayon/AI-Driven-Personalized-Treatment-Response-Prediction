from huggingface_hub import snapshot_download

# আপনার Hugging Face টোকেনটি এখানে নিচের কোটেশনের ভেতরে বসান
my_token = "আপনার_টোকেন_এখানে_দিন" 

try:
    print("ডাউনলোড শুরু হচ্ছে... ধৈর্য ধরুন।")
    snapshot_download(
        repo_id="meta-llama/Llama-3.2-1B",
        local_dir=".",
        token=my_token
    )
    print("অভিনন্দন! মডেলটি সফলভাবে ডাউনলোড হয়েছে।")
except Exception as e:
    print(f"একটি সমস্যা হয়েছে: {e}")