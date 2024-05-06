import datasets as ds
from datasets import load_dataset
from transformers import AutoTokenizer


def define_dataset_parameters():
    # Dataset parameters
    dataset_parameters = {}
    dataset_parameters["dataset_full_name"] = "midas/inspec"
    dataset_parameters["dataset_subset"] = "raw"
    dataset_parameters["dataset_document_column"] = "document"

    dataset_parameters["keyphrase_sep_token"] = ";"

    return dataset_parameters


def load_corpus(dataset_full_name, dataset_subset):
    dataset = load_dataset(dataset_full_name, dataset_subset)
    return dataset


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART", add_prefix_space=True)
    return tokenizer


def preprocess_keyphrases(text_ids, kp_list):
    tokenizer = load_tokenizer()
    kp_order_list = []
    kp_set = set(kp_list)
    text = tokenizer.decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    text = text.lower()
    for kp in kp_set:
        kp = kp.strip()
        kp_index = text.find(kp.lower())
        kp_order_list.append((kp_index, kp))

    kp_order_list.sort()
    present_kp, absent_kp = [], []

    for kp_index, kp in kp_order_list:
        if kp_index < 0:
            absent_kp.append(kp)
        else:
            present_kp.append(kp)
    return present_kp, absent_kp


def preprocess_fuction(samples):
    dataset_parameters = define_dataset_parameters()
    dataset_full_name = dataset_parameters["dataset_full_name"]
    dataset_subset = dataset_parameters["dataset_subset"]
    dataset_document_column = dataset_parameters["dataset_document_column"]
    keyphrase_sep_token = dataset_parameters["keyphrase_sep_token"]
    tokenizer = load_tokenizer()
    processed_samples = {"input_ids": [], "attention_mask": [], "labels": []}
    for i, sample in enumerate(samples[dataset_document_column]):
        input_text = " ".join(sample)
        inputs = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
        )
        present_kp, absent_kp = preprocess_keyphrases(
            text_ids=inputs["input_ids"],
            kp_list=samples["extractive_keyphrases"][i]
                    + samples["abstractive_keyphrases"][i],
        )
        keyphrases = present_kp
        keyphrases += absent_kp

        target_text = f" {keyphrase_sep_token} ".join(keyphrases)

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                target_text, max_length=40, padding="max_length", truncation=True
            )
            targets["input_ids"] = [
                (t if t != tokenizer.pad_token_id else -100)
                for t in targets["input_ids"]
            ]
        for key in inputs.keys():
            processed_samples[key].append(inputs[key])
        processed_samples["labels"].append(targets["input_ids"])
    return processed_samples
