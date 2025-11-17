# === STRATIFIED SPLIT 80-10-10 (SAMA SEPERTI BASELINE) ===
# Dapatkan semua labels dari dataset
all_labels = []
for i in range(len(dataset)):
    _, _, label = dataset[i]
    all_labels.append(label.item() if torch.is_tensor(label) else label)

all_labels = np.array(all_labels)

# First split: train vs temp (80% vs 20%)
train_indices, temp_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    random_state=seed,
    stratify=all_labels
)

# Second split: val vs test (dari 20% temp, bagi menjadi 10% val dan 10% test)
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    random_state=seed,
    stratify=[all_labels[i] for i in temp_indices]
)
