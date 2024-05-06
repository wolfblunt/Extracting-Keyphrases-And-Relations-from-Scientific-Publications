import datasets as ds
from datasets import load_dataset
from transformers import AutoTokenizer
import sys
from sklearn.metrics import classification_report
from data_preprocessing import load_dataset, preprocess_fuction, define_dataset_parameters
from t5_small_inspec import KeyphraseGenerationPipeline
from kbir_inspec import KBirKeyphraseExtractionPipeline
from keybart_inspec import KeyBartKeyphraseGenerationPipeline
from distilbert_inspec import DistilbertKeyphraseExtractionPipeline


def tag_bio(text, ann):
    tags = ['O'] * len(text)
    last_i = 0
    for key_phrase in ann:

        for i in range(len(text)):

            if tags[i] == 'O':
                c = 0
                for count in range(len(key_phrase)):
                    if i + count < len(text):
                        if key_phrase[count] != text[i + count]:
                            break
                        else:
                            c = c + 1

                if c == len(key_phrase):
                    tags[i] = 'B'
                    for k in range(1, len(key_phrase)):
                        tags[i + k] = 'I'
                    last_i = i + c
    return tags


def get_lower(text, ann):
    return [[j.lower() for j in i.split()] for i in ann], [i.lower() for i in text.split()]


def preprocess(text, ann):
    return tag_bio(text, ann)


def helper():
    data_document = []
    train = tokenized_dataset['train']
    test = tokenized_dataset['test']
    val = tokenized_dataset['validation']

    train_doc = {}
    train_doc["train_document"] = [" ".join(i['document']) for i in train]
    train_doc["train_keyphrase"] = [i['extractive_keyphrases'] for i in train]
    train_doc["train_tags"] = train['doc_bio_tags']

    test_doc = {}
    test_doc["test_document"] = [" ".join(i['document']) for i in test]
    test_doc["test_keyphrase"] = [i['extractive_keyphrases'] for i in test]
    test_doc["test_tags"] = test['doc_bio_tags']

    val_doc = {}
    val_doc["val_document"] = [" ".join(i['document']) for i in val]
    val_doc["val_keyphrase"] = [i['extractive_keyphrases'] for i in val]
    val_doc["val_tags"] = val['doc_bio_tags']

    data_document.append(train_doc)
    data_document.append(test_doc)
    data_document.append(val_doc)

    return data_document


def report(document, test_tags, keyphrases):
    y_pred, y_true = [], []
    for i in range(len(document)):
        key, text = get_lower(document[i], keyphrases[i])

        tag = preprocess(text, key)
        y_pred += tag
        y_true += test_tags[i]

    print(classification_report(y_true, y_pred))


def model_func(model):
    model_name = None
    # generator = None
    if model == 't5':
        model_name = "ml6team/keyphrase-generation-t5-small-inspec"
        generator = KeyphraseGenerationPipeline(model=model_name)
        return model_name, generator
    if model == 'distilbert':
        model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
        extractor = DistilbertKeyphraseExtractionPipeline(model=model_name)
        return model_name, extractor
    if model == 'kbir':
        model_name = "ml6team/keyphrase-extraction-kbir-inspec"
        extractor = KBirKeyphraseExtractionPipeline(model=model_name)
        return model_name, extractor
    # if model_name == 'keybart':
    model_name = "ml6team/keyphrase-generation-keybart-inspec"
    generator = KeyBartKeyphraseGenerationPipeline(model=model_name)

    return model_name, generator


def generate_keyphrases(model, test_document):
    if model == 't5' or model == 'keybart':
        model_name, generator = model_func(model)
        keyphrases = generator(test_document["test_document"])
    else:
        model_name, extractor = model_func(model)
        keyphrases = extractor(test_document["test_document"])
    return keyphrases


if __name__ == '__main__':
    # model_name = "ml6team/keyphrase-generation-t5-small-inspec"
    model = input("Enter model name : ")
    # model_name, generator = model_func(model)
    dataset_parameters = define_dataset_parameters()
    dataset_full_name = dataset_parameters["dataset_full_name"]
    dataset_subset = dataset_parameters["dataset_subset"]

    # Load dataset
    dataset = load_dataset(dataset_full_name, dataset_subset)

    # Preprocess dataset
    tokenized_dataset = dataset.map(preprocess_fuction, batched=True)

    # generator = KeyphraseGenerationPipeline(model=model_name)

    data_docs = helper()

    train_doc = data_docs[0]
    test_doc = data_docs[1]
    val_doc = data_docs[2]

    keyphrases = generate_keyphrases(model, test_doc)
    # keyphrases = generator(test_doc["test_document"])

    report(test_doc["test_document"], test_doc["test_tags"], keyphrases)
