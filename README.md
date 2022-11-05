# FRATIS

This repository contains the translated transcription of the ATIS dataset. Only the sentences are translated, the tokens remains unmodified.

The used code is inspired by this notebook https://www.kaggle.com/code/siddhadev/atis-dataset-clean-re-split-kernel/notebook.

The translation has been realized using Google Translate throught [Deep-Translator](https://github.com/nidhaloff/deep-translator).

## Running resplit and translation script

```bash
$ python resplit_and_translate.py
len(atis_all) = 21708
len(atis_uniques) = 10948
Translating "train" subset (1/3):
100%|████████████████████████████████████████████████████████████████████| 8671/8671 [1:02:30<00:00,  2.31it/s] 
Translating "dev" subset (2/3):
100%|██████████████████████████████████████████████████████████████████████| 1152/1152 [05:36<00:00,  3.43it/s] 
Translating "test" subset (3/3):
100%|██████████████████████████████████████████████████████████████████████| 1125/1125 [04:24<00:00,  4.25it/s]
```
