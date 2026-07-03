import json
test_path = "./data/AWS_SAM_split/test.json"
test_path = "./data/AWS_SAM_split/train.json"

with open(test_path, 'r') as f:
    test_data = json.load(f)

print(test_data.keys())

print(f"annotations: {len(test_data['annotations'])}")