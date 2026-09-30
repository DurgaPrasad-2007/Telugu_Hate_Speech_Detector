# Data Quality Report

Computed on internal label convention (1=Abusive) after RAW_LABEL_MAP.


## telugu

| split | n | abusive | non_abusive | char_len_mean | dup_texts | emoji | url |
|---|---|---|---|---|---|---|---|
| train | 23980 | 12401 | 11579 | 85.7 | 1 | 9624 | 1 |
| val | 3000 | 1618 | 1382 | 87.6 | 0 | 1216 | 0 |
| test | 3000 | 1564 | 1436 | 85.1 | 0 | 1180 | 0 |

Script mix (train): {'telugu_script': 18959, 'telugu+english': 5021}

Leakage (exact text overlap): {'train_val_overlap': 2, 'train_test_overlap': 4, 'val_test_overlap': 1}


## codemixed

| split | n | abusive | non_abusive | char_len_mean | dup_texts | emoji | url |
|---|---|---|---|---|---|---|---|
| train | 3179 | 1542 | 1637 | 67.3 | 8 | 3 | 0 |
| val | 398 | 194 | 204 | 66.6 | 1 | 1 | 0 |
| test | 400 | 194 | 206 | 64.1 | 0 | 0 | 0 |

Script mix (train): {'latin_only': 1746, 'telugu_script': 1113, 'telugu+english': 318, 'other': 2}

Leakage (exact text overlap): {'train_val_overlap': 6, 'train_test_overlap': 11, 'val_test_overlap': 2}
