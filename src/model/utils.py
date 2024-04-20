import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu


def corpus_bleu(references, hypotheses):
    """
    Calculate the BLEU score for a corpus of references and hypotheses.

    Args:
        references (List[List[str]]): A list of reference sentences, where each sentence is a list of tokens.
        hypotheses (List[List[str]]): A list of hypothesis sentences, where each sentence is a list of tokens.

    Returns:
        float: The BLEU score.
    """
    return nltk_corpus_bleu(references, hypotheses)

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs, checkpoint_path):
    """
    Train the model with early stopping and save checkpoints.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (nn.Module): The loss function.
        num_epochs (int): Number of epochs to train.
        checkpoint_path (str): Path to save the checkpoints.
    """
    best_bleu_score = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (en_input_batch, vi_input_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(en_input_batch, vi_input_batch)

            # Compute loss
            loss = criterion(output.view(-1, output.size(-1)), vi_input_batch.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / 50
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                total_loss = 0

        # Validation
        model.eval()
        references, hypotheses = [], []

        with torch.no_grad():
            for en_input_batch, vi_input_batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                output = model(en_input_batch, vi_input_batch)
                output_ids = torch.argmax(output, dim=-1)
                output_ids = output_ids.cpu().numpy().tolist()

                # Convert ids to tokens
                references.extend([[str(idx) for idx in sent] for sent in vi_input_batch.cpu().numpy().tolist()])
                hypotheses.extend([str(idx) for idx in sent] for sent in output_ids)

        # Calculate BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        print(f"Validation BLEU Score: {bleu_score:.4f}")

        # Save the model checkpoint if the BLEU score improves
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            torch.save(model.state_dict(), checkpoint_path.replace(".pth", "_best.pth"))
            print("Saved best checkpoint.")

        # Save the model checkpoint at the end of training
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved final checkpoint.")

    print("Training finished!")

def test(model, test_loader):
    """
    Evaluate the model on the test set and calculate the BLEU score.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
    """
    model.eval()
    references, hypotheses = [], []

    with torch.no_grad():
        for en_input_batch, vi_input_batch in tqdm(test_loader, desc="Testing"):
            output = model(en_input_batch, vi_input_batch)
            output_ids = torch.argmax(output, dim=-1)
            output_ids = output_ids.cpu().numpy().tolist()

            # Convert ids to tokens
            references.extend([[str(idx) for idx in sent] for sent in vi_input_batch.cpu().numpy().tolist()])
            hypotheses.extend([str(idx) for idx in sent] for sent in output_ids)

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"Test BLEU Score: {bleu_score:.4f}")