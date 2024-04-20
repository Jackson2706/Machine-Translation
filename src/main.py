from fire import Fire
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch import nn
from torch import optim as opt

from config import get_data_from_json
from dataset import EnglishVietNamDataset
from model import *
from utils import load_dict_from_pickle

def train(config_path):
    config_data = get_data_from_json(config_path)
    en_tokenizer = get_tokenizer("basic_english")
    vi_tokenizer = get_tokenizer("basic_english")
    train_dataset = EnglishVietNamDataset(data_folder=config_data["TRAIN_FOLDER_PATH"],
                                          phase=config_data["TRAIN_PHASE_NAME"],
                                          en_tokenizer=en_tokenizer,
                                          vi_tokenizer=vi_tokenizer,
                                          max_sequence_length= config_data["MAX_SEQUENCE_LENGTH"],
                                          threshold=config_data["THRESHOLD"],
                                          en_vocab_save_path=config_data["EN_VOCAB_PATH"],
                                          vi_vocab_save_path=config_data["VI_VOCAB_PATH"],
                                          en2vi=config_data["ENGLISH_TO_VIETNAMESE"]
                                          )    
    
    valid_dataset = EnglishVietNamDataset(data_folder=config_data["VALID_FOLDER_PATH"],
                                          phase=config_data["VALID_PHASE_NAME"],
                                          en_tokenizer=en_tokenizer,
                                          vi_tokenizer=vi_tokenizer,
                                          max_sequence_length= config_data["MAX_SEQUENCE_LENGTH"],
                                          threshold=config_data["THRESHOLD"],
                                          en2vi=config_data["ENGLISH_TO_VIETNAMESE"]
                                          )  

    train_loader =   DataLoader(dataset=train_dataset,
                                batch_size=config_data["TRAIN_BATCH_SIZE"],
                                shuffle=True,
                                drop_last=True)
    
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=config_data["VALID_BATCH_SIZE"],
                              shuffle=False,
                              drop_last=False)
    en_vocab = load_dict_from_pickle(config_data["EN_VOCAB_PATH"])
    vi_vocab = load_dict_from_pickle(config_data["VI_VOCAB_PATH"])
    if config_data["MODEL_NAME"] == "RNN":
        if config_data["ENGLISH_TO_VIETNAMESE"]: 
            encoder = RNNEncoder(len(en_vocab), config_data["EMBEDDING_DIM"], config_data["HIDDEND_DIM"])
            decoder = RNNDecoder(len(vi_vocab), config_data["EMBEDDING_DIM"], config_data["HIDDEND_DIM"])
        else: 
            encoder = RNNEncoder(len(vi_vocab), config_data["EMBEDDING_DIM"], config_data["HIDDEND_DIM"])
            decoder = RNNDecoder(len(en_vocab), config_data["EMBEDDING_DIM"], config_data["HIDDEND_DIM"])
        model = RNN_Seq2Seq_Model(encoder=encoder, decoder=decoder)
    
    elif config_data["MODEL_NAME"] == "TRANSFORMER":
        if config_data["ENGLISH_TO_VIETNAMESE"]: 
            encoder = Transformer_Encoder(len(en_vocab), config_data["EMBEDDING_DIM"], config_data["MODEL_DIM"], config_data["NUM_HEAD"])
            decoder = Transformer_Decoder(len(vi_vocab), config_data["EMBEDDING_DIM"], config_data["MODEL_DIM"], config_data["NUM_HEAD"], config_data["MAX_SEQUENCE_LENGTH"])
        else: 
            encoder = Transformer_Encoder(len(vi_vocab), config_data["EMBEDDING_DIM"], config_data["MODEL_DIM"], config_data["NUM_HEAD"])
            decoder = Transformer_Decoder(len(en_vocab), config_data["EMBEDDING_DIM"], config_data["MODEL_DIM"], config_data["NUM_HEAD"], config_data["MAX_SEQUENCE_LENGTH"])
        model = Transformer_Seq2Seq_Model(encoder=encoder, decoder=decoder)
    else:
        return
    
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(), config_data["LEARNING_RATE"])
    train_utils(model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=config_data["NUM_EPOCH"],
                checkpoint_path=config_data["CHECKPOINT_PATH"])
def test(config_path):
    pass


if __name__ == "__main__":
    Fire()