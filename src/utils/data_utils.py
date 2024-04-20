from langid.langid import LanguageIdentifier, model
import re
import string
from torchtext.vocab import build_vocab_from_iterator
import pickle


def load_dict_from_pickle(pickle_file):
    """
    Load a dictionary from a pickle file.

    Args:
        pickle_file (str): Path to the pickle file.

    Returns:
        dict: The loaded dictionary.
    """
    with open(pickle_file, "rb") as file:
        data_dict = pickle.load(file)
    
    return data_dict

def save_dict_to_pickle(data_dict, pickle_file):
    """
    Save a dictionary to a pickle file.

    Args:
        data_dict (dict): The dictionary to be saved.
        pickle_file (str): Path to the pickle file.
    """
    with open(pickle_file, "wb") as file:
        pickle.dump(data_dict, file)


# Loại bỏ các mẫu có chứa kí tư không phải là tiếng anh
def identify_lang(en_data, vi_data, threshold: float = 0.9):
  identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
  en_list = []
  vi_list = []
  for en_sentence, vi_sentence in zip(en_data, vi_data):
    en_score = identifier.classify(en_sentence)
    vi_score = identifier.classify(vi_sentence)
    if (en_score[0] == 'en' and en_score[1] >= threshold) and (vi_score[0] == 'vi' and vi_score[1] >= threshold):
      en_list.append(en_sentence)
      vi_list.append(vi_sentence)
  return en_list, vi_list


# Tiền xử lý data
def preprocessing_text(text: str):
    """
    Preprocesses text by removing special patterns, punctuation, digits, and emojis.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: Clean text containing only Vietnamese characters.
    """
    # Define the URL pattern
    url_pattern = re.compile(r"https?://\s+\www\.\s+")
    # Replace URLs with whitespace
    text = url_pattern.sub(r" ", text)

    # Define the HTML pattern
    html_pattern = re.compile(r"<[^<>]+>")
    # Replace HTML patterns with whitespace
    text = html_pattern.sub(" ", text)

    # Define punctuation and digits pattern
    replace_chars = list(string.punctuation + string.digits)
    # Replace punctuation and digits with whitespace
    for char in replace_chars:
        text = text.replace(char, " ")

    # Define the emoji pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F1F2-\U0001F1F4"  # Macau flag
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F600-\U0001F64F"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U0001F1F2"
        "\U0001F1F4"
        "\U0001F620"
        "\u200d"
        "\u2640-\u2642"
        "]+",
        flags=re.UNICODE,
    )
    # Replace emojis with whitespace
    text = emoji_pattern.sub(r" ", text)

    # Remove duplicate whitespace
    text = re.sub(r'\s+', ' ', text)

    # Return lowercase text
    return text.lower()

def preprocess(data_list):
  process_data_list = []
  for sentence in data_list:
    process_data_list.append(preprocessing_text(sentence))
  return process_data_list



def yield_tokens(sentences, tokenizer):
  for sentence in sentences:
    yield tokenizer(sentence)



def build_vocabulary(sentences, tokenizer):
  vocabulary = build_vocab_from_iterator(yield_tokens(sentences, tokenizer),  specials=["<unk>", "<pad>", "<sos>", "<eos>"])
  vocabulary.set_default_index(vocabulary["<unk>"])
  return vocabulary