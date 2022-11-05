import json
import os


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, 'data')
DATA_CLEANED_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, 'data_cleaned')
DATA_TRANSLATED_DIRECTORY = os.path.join(
    SCRIPT_DIRECTORY, 'data_cleaned_and_translated')


def ensure_directory(path: str) -> bool:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception:
            import traceback
            print(traceback.format_exc())

    return os.path.exists(path) and not os.path.isfile(path)


def load_dataset_files(filenames: str | list[str] | tuple[str]) -> dict:
    data_subsets = {}

    if isinstance(filenames, str):
        filenames = [filenames]

    for name in filenames:
        dataset_content = []
        dataset_name = os.path.basename(name[:name.rfind('.')])
        # print(file_content[:30])
        for line in open(name, 'r'):
            # Extract tokens, etc. for each line of the .iob file
            tokens, si = map(str.split, line.split("\t"))
            slots, intent = si[:-1] + ['O'], si[-1]

            # Break everything if the is any inconsistency in the dataset
            assert len(tokens) == len(slots)

            # Add the data to dataset representation
            dataset_content.append((tokens, slots, intent))

        # Add each dataset to the dictionary
        data_subsets.update({dataset_name: dataset_content})

    return data_subsets

# TODO: Make the function clearer


def split_atis(ads, split=[0.8, 0.1, 0.1], random_state=None):
    """ Splits the ATIS dataset by starting with the least common labels."""
    from random import Random
    from collections import defaultdict, Counter
    import numpy as np
    assert sum(split) == 1
    random = None if random_state is None else Random(random_state)

    resulting_split = [[] for _ in split]
    used = set()  # used samples
    slot_to_samples = defaultdict(set)  # slot to sample
    intent_to_samples = defaultdict(set)  # intent to sample
    for index, (_, slots, intent) in enumerate(ads):  # build label to sample maps
        intent_to_samples[intent].add(index)
        for slot in set(slots):
            slot_to_samples[slot].add(index)

    # sort according to usage
    s2f = Counter({slot: len(indexes)
                  for slot, indexes in slot_to_samples.items()}).most_common()
    i2f = Counter({intent: len(indexes)
                  for intent, indexes in intent_to_samples.items()}).most_common()

    while True:
        # select the least common intent or slot label
        if len(i2f) < 1 and len(s2f) < 1:            # both empty
            break
        elif len(i2f) > 0 and len(s2f) > 0:          # both non empty
            use_intent = i2f[-1][1] < s2f[-1][1]
        else:
            use_intent = len(i2f) > 0

        # get the samples of least common label
        if use_intent:
            intent, _ = i2f.pop()
            indexes = list(intent_to_samples.pop(intent))
        else:
            slot, _ = s2f.pop()
            indexes = list(slot_to_samples.pop(slot))

        # shuffle the samples of the selected label
        if random is not None:
            random.shuffle(indexes)

        splits = [[] for _ in split]
        # put each sample in a split maintaining the split ratios
        split = np.array(split)
        for index in indexes:
            lens = np.array([len(sp) for sp in splits])
            # fill the split with the highest frequency (reverse ratio) offset
            sndx = np.argmax(1/(lens/(lens.sum()+1e-12)+1e-12) - 1/split)
            if index not in used:
                used.add(index)
                splits[sndx].append(index)

        # add the splitted label samples to the result split
        for index, indexes in enumerate(splits):
            resulting_split[index].extend(
                np.array(atis_clean, dtype=object)[indexes].tolist())

    return resulting_split


def translate_dataset(
    dataset: dict,
    translator: str = 'google',
    source_language: str = 'en',
    target_language: str = 'fr'
) -> dict:
    from tqdm import tqdm as progress_bar
    from deep_translator.engines import __engines__
    translator = __engines__[translator](
        source=source_language, target=target_language)

    def translate(text: str) -> str:
        return translator.translate(text)

    translated_dataset = {}
    number_of_subset = len(list(dataset.keys()))

    for index, (subset_name, data) in enumerate(dataset.items(), start=1):
        print(f'Translating "{subset_name}" subset ({index}/{number_of_subset}):')

        translated_subset = []
        for element in progress_bar(data):
            # Extract information (tokens, slots, intent)
            tokens, _, intent = element
            sentence = ' '.join(tokens[1:-1])
            translated_sentence = translate(sentence)
            translated_tokens = ['BOS'] + \
                translated_sentence.split(' ') + ['EOS']
            empty_slots = ['O' for _ in translated_tokens]
            translated_subset.append((translated_tokens, empty_slots, intent))

        translated_dataset.update({subset_name: translated_subset})

    return translated_dataset


if __name__ == '__main__':
    # Create missing directories (if needed)
    ensure_directory(DATA_CLEANED_DIRECTORY)
    ensure_directory(DATA_TRANSLATED_DIRECTORY)

    # List all the dataset files in data
    def endswith(string: str, substring: str = '.iob') -> bool:
        return string.endswith(substring)

    dataset_files = filter(endswith, os.listdir(DATA_DIRECTORY))
    dataset_files = [os.path.join(DATA_DIRECTORY, filename)
                     for filename in dataset_files]

    # Load the data and merge all the data
    atis = load_dataset_files(dataset_files)
    atis_all = []
    for _, subset in atis.items():
        atis_all += subset

    # Remove the duplicates
    atis_uniques = set([(tuple(tokens), tuple(slots), intent)
                        for tokens, slots, intent in atis_all])
    # Convert back from tuples to lists
    atis_clean = [(list(tokens), list(slots), intent)
                  for tokens, slots, intent in sorted(atis_uniques)]

    print(f'{len(atis_all) = }\n{len(atis_uniques) = }')

    # Resplit dataset (train, dev/validation, test)
    atis_resplit = split_atis(atis_clean, [0.8, 0.1, 0.1], random_state=42)
    # print(f'{atis_resplit[1]}')
    atis_resplit = {name: value for name, value in zip(
        ('train', 'dev', 'test'), atis_resplit)}

    # Save resplited dataset
    json.dump(atis_resplit, open(os.path.join(
        DATA_CLEANED_DIRECTORY, 'dataset.json'), 'w'), indent=4)

    # Translate dataset
    atis_translated = translate_dataset(atis_resplit)

    # Save translated dataset
    json.dump(atis_translated, open(os.path.join(
        DATA_TRANSLATED_DIRECTORY, 'dataset.json'), 'w'), indent=4)
