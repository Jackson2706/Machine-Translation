from torch.utils.data import Dataset
import torch
import os
from  torchtext.data.utils import get_tokenizer


from utils.data_utils import preprocess, build_vocabulary, identify_lang, save_dict_to_pickle

class EnglishVietNamDataset(Dataset):
  def __init__(self, data_folder, phase, en_tokenizer, vi_tokenizer ,max_sequence_length, threshold, en_vocab_save_path=None, vi_vocab_save_path=None, en2vi = True):
    en_data_file_path = os.path.join(data_folder, f"{phase}.en")
    vi_data_file_path = os.path.join(data_folder, f"{phase}.vi")

    raw_en_data = []
    raw_vi_data = []
    with open(en_data_file_path, "r") as file:
      for line in file:
        raw_en_data.append(line)

    with open(vi_data_file_path, "r") as file:
      for line in file:
        raw_vi_data.append(line)
    clean_en_data, clean_vi_data = identify_lang(raw_en_data, raw_vi_data, threshold)
    self.input_en_data = preprocess(clean_en_data)
    self.input_vi_data = preprocess(clean_vi_data)
    self.en_vocab = build_vocabulary(self.input_en_data, en_tokenizer)
    self.vi_vocab = build_vocabulary(self.input_vi_data, vi_tokenizer)
    if en_vocab_save_path and vi_vocab_save_path:
        save_dict_to_pickle(self.en_vocab, en_vocab_save_path)
        save_dict_to_pickle(self.vi_vocab, vi_vocab_save_path)
    self.en_tokenizer = en_tokenizer
    self.vi_tokenizer = vi_tokenizer
    self.max_sequence_length = max_sequence_length
    self.en2vi = en2vi

  def __len__(self):
    return len(self.input_en_data)

  def __getitem__(self, index):
    en_data = self.input_en_data[index]
    vi_data = self.input_vi_data[index]
    if self.en2vi:
      en_vectorize_data = self._vectorize(en_data, self.en_tokenizer, self.en_vocab, self.max_sequence_length)
      vi_vectorize_data = self._vectorize(vi_data, self.vi_tokenizer, self.vi_vocab, self.max_sequence_length, True)
      en_tensor = torch.tensor(en_vectorize_data, dtype=torch.long)
      vi_tensor = torch.tensor(vi_vectorize_data, dtype=torch.long)
      return en_tensor, vi_tensor
    else:
      en_vectorize_data = self._vectorize(en_data, self.en_tokenizer, self.en_vocab, self.max_sequence_length, True)
      vi_vectorize_data = self._vectorize(vi_data, self.vi_tokenizer, self.vi_vocab, self.max_sequence_length)
      en_tensor = torch.tensor(en_vectorize_data, dtype=torch.long)
      vi_tensor = torch.tensor(vi_vectorize_data, dtype=torch.long)
      return en_tensor, vi_tensor

  def _vectorize(self, text, tokenizer, vocab, max_sequence_length, add_sos = False):
    tokens = tokenizer(text)
    tokens = [vocab[token] for token in tokens] + [vocab["<eos>"]]
    if add_sos:
      tokens = [vocab["<sos>"]] + tokens
    token_ids = tokens[:max_sequence_length] + [vocab["<pad>"]] * max((max_sequence_length - len(tokens)), 0)
    return token_ids


if __name__ == "__main__":
  # Test

    en_tokenizer = get_tokenizer('basic_english')
    vi_tokenizer = get_tokenizer('basic_english')

    train_dataset = EnglishVietNamDataset(data_folder="../data/train-en-vi",
                                        phase="train",
                                        en_tokenizer=en_tokenizer,
                                        vi_tokenizer=vi_tokenizer,
                                        max_sequence_length=50,
                                        threshold=0.95,
                                        en2vi=True)

    print(train_dataset[56])