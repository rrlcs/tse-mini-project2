import json
from torch.utils.data import DataLoader, IterableDataset


class JsonDataset(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    # print("sam: ", sample["constraint"])
                    yield sample["constraint"], sample["program"]


dataset = JsonDataset(["dataset.json"])
dataloader = DataLoader(dataset, batch_size=32)
# print(dataloader.dataset.__getitem__(10))
for batch in dataloader:
    print(batch[0].get("constraint"))
#     y = model(batch)
