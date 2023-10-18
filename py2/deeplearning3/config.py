cfg = {
    "model": {
        "struct": 50,
        "model_path": None,
        # "save_path": "output/ResNet/ResNet"
        "save_path": None
    },
    "num_class": 2,
    "train": {
        "root": "work/palm/PALM-Training400/PALM-Training400",
        "batch_size": 10
    },
    "val": {
        "root": "work/palm/PALM-Validation400",
        "csvfile": "labels.csv",
        "batch_size": 10
    },
    "lr": 0.0001,
    "step": 800
}