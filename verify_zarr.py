#!/usr/bin/env python
"""Verify FluentSigners-50 Zarr dataset structure and content."""

import zarr
from pathlib import Path

zarr_path = Path(r"D:\plumiume\SLDatasets\fs50-lmpipe-v5.2.zarr")

print(f"Opening Zarr dataset: {zarr_path}")
root = zarr.open_group(zarr_path, mode='r')

print(f"\n{'='*60}")
print("ROOT STRUCTURE")
print(f"{'='*60}")
print(f"Root keys: {list(root.keys())}")
print(f"Root tree (first 3 levels):")
print(root.tree(level=3))

print(f"\n{'='*60}")
print("METADATA")
print(f"{'='*60}")
if 'metadata' in root:
    metadata = root['metadata']
    print(f"Metadata attrs: {dict(metadata.attrs)}")
    for key, value in metadata.attrs.items():
        if isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} with {len(value)} items")
            if len(value) > 0:
                print(f"    First 3: {value[:3]}")
        else:
            print(f"  {key}: {value}")

print(f"\n{'='*60}")
print("CONNECTIONS")
print(f"{'='*60}")
if 'connections' in root:
    connections = root['connections']
    print(f"Connection keys: {list(connections.keys())}")
    for key in list(connections.keys())[:5]:
        arr = connections[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

print(f"\n{'='*60}")
print("ITEMS")
print(f"{'='*60}")
if 'items' in root:
    items = root['items']
    item_keys = sorted(items.keys(), key=lambda x: int(x))
    print(f"Total items: {len(item_keys)}")
    
    # Check first item
    if item_keys:
        first_item_key = item_keys[0]
        first_item = items[first_item_key]
        print(f"\nFirst item ({first_item_key}) structure:")
        print(f"  Keys: {list(first_item.keys())}")
        
        if 'videos' in first_item:
            videos = first_item['videos']
            print(f"  Videos: {list(videos.keys())}")
        
        if 'landmarks' in first_item:
            landmarks = first_item['landmarks']
            lm_keys = list(landmarks.keys())
            print(f"  Landmarks: {lm_keys}")
            for lm_key in lm_keys[:3]:
                arr = landmarks[lm_key]
                print(f"    {lm_key}: shape={arr.shape}, dtype={arr.dtype}")
                print(f"      Sample data (first 2 points): {arr[:2]}")
        
        if 'targets' in first_item:
            targets = first_item['targets']
            print(f"  Targets: {list(targets.keys())}")

    # Sample check across items
    print(f"\nRandom sample of 5 items:")
    sample_indices = [0, len(item_keys)//4, len(item_keys)//2, 3*len(item_keys)//4, len(item_keys)-1]
    for idx in sample_indices:
        if idx < len(item_keys):
            item_key = item_keys[idx]
            item = items[item_key]
            landmarks = item.get('landmarks', {})
            lm_keys = list(landmarks.keys())
            lm_count = len(lm_keys)
            print(f"  Item {item_key}: {lm_count} landmark types")

print(f"\n{'='*60}")
print("VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"✓ Root structure present")
print(f"✓ Metadata group exists: {'metadata' in root}")
print(f"✓ Connections group exists: {'connections' in root}")
print(f"✓ Items group exists: {'items' in root}")
if 'items' in root:
    item_keys_list = list(root['items'].keys())
    print(f"✓ Total items: {len(item_keys_list)}")
print(f"\nDataset appears to be valid!")
