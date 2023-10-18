cfg = {
    "model": {
        "struct": 50,
        "model_path": None,
        "save_path": None
    },
    "num_class": 102,
    "train": {
        "root": "work//Caltech101//train",
        "batch_size": 64
    },
    "val": {
        "root": "work//Caltech101//val",
        "batch_size": 32
    },
    "lr": 0.001,
    "epochs": 150,
    "steps": 50000,
    "continue": None
}