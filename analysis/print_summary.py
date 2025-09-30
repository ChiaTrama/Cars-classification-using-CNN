import pickle
import sys

'''
Prints the summary of a training run from a pickle file.
'''
summary_path = "runs/ResNet18_FineTuning_224x224/summary_make_crossentropy_bs256_ep45.pkl"
#summary_path = "runs/SiameseResNet18_FineTuning_224x224/summary_make_contrastive_bs128_ep20_verification_easy.pkl"
#summary_path = "runs/InceptionV3_299x299/summary_make_crossentropy_bs64_ep45.pkl"
with open(summary_path, "rb") as f:
    summary = pickle.load(f)
if isinstance(summary, list):
    summary = summary[0]

print("=== SUMMARY ===")
for k, v in summary.items():
    if isinstance(v, list):
        print(f"{k}: [{', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in v)}]")
    else:
        print(f"{k}: {v}")