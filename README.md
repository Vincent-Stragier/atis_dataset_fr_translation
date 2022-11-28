# FRATIS

This repository contains the translated transcription of the ATIS dataset. Only the sentences are translated, the tokens remains unmodified.

The used code is inspired by [this notebook](https://www.kaggle.com/code/siddhadev/atis-dataset-clean-re-split-kernel/notebook).

The translation has been realized using Google Translate throught [Deep-Translator](https://github.com/nidhaloff/deep-translator).

## Running resplit and translation script

```bash
$ python resplit_and_translate.py
len(atis_all) = 10849
len(atis_uniques) = 5474
Translating "train" subset (1/3):
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4305/4305 [18:01<00:00,  3.98it/s]
Translating "dev" subset (2/3):
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 595/595 [07:47<00:00,  1.27it/s] 
Translating "test" subset (3/3):
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 574/574 [07:25<00:00,  1.29it/s] 
```
