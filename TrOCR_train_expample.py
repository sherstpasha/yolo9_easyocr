from tools import train_trocr



train_trocr(
    train_csv=r"C:\Users\user\Desktop\all_data_small\new_dataset\labels.csv",
    train_root=r"C:\Users\user\Desktop\all_data_small\new_dataset",
    val_csv=r"C:\Users\user\Desktop\all_data_small\new_val\labels.csv",
    val_root=r"C:\Users\user\Desktop\all_data_small\new_val",
    model_name=r"C:\Users\user\TrOCR\trocr-base-ru",
    output_dir=r"C:\Users\user\TrOCR\output",
    epochs=100,
    batch_size=15,
    learning_rate=5e-5,
    gpu=True
)