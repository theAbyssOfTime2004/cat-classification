from pathlib import Path
import shutil
from collections import Counter

# Count images per breed
counts = Counter()
for f in Path('data/processed/train').rglob('*.jpg'):
    counts[f.parent.name] += 1

# Get top 20
top20 = [breed for breed, _ in counts.most_common(20)]
print("Top 20 breeds:", top20)
print(f"\nKeeping {len(top20)} breeds, removing {len(counts) - len(top20)} breeds")

# Delete breeds that are not in top 20
for split in ['train', 'val', 'test']:
    split_path = Path(f'data/processed/{split}')
    if split_path.exists():
        for breed_dir in split_path.iterdir():
            if breed_dir.is_dir() and breed_dir.name not in top20:
                print(f"Deleting {breed_dir}")
                shutil.rmtree(breed_dir)

print("✓ Done! Only top 20 breeds remain in data/processed/")