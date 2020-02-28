import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

from tqdm import tqdm_notebook as tqdm
from utils import smoothing


def check_accuracy(loader, model, device, max_sample_size=None, preprocessor=None):
    """
    Check the accuracy over a validation dataset. This is used during training
    to check if the model is overfitting.
    loader: pytorch Dataloader used to check the accuracy
    model: BERThoven model on which the accuracy is checked
    device: GPU or CPU
    max_sample_size: maximum length of a sentence
    preprocessor: sklearn.preprocessing method. If the model is trained with 
    preprocessing, the same preprocessor should be added
    """

    model = model.to(device=device)
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0

    with torch.no_grad():
        scores_epoch = []
        truth_epoch = []

        for x1, x1_mask, x2, x2_mask, y in tqdm(loader, "Checking accuracy", leave=False):
            truth_epoch += y.tolist()
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            scores = model.forward((x1, x1_mask), (x2, x2_mask))

            scores = scores.cpu().numpy().reshape(-1, 1)
            y = y.cpu().numpy().reshape(-1, 1)

            if preprocessor is not None:
                scores = preprocessor.inverse_transform(scores)
                y = preprocessor.inverse_transform(y)
            scores_epoch += scores.reshape(-1).tolist()

            abs_error += np.abs(scores - y).sum().item()
            sqr_error += ((scores - y) ** 2).sum().item()
            num_samples += scores.shape[0]
            if max_sample_size != None and num_samples >= num_samples:
                break
        rmse = (sqr_error / num_samples) ** 0.5
        mae = abs_error / num_samples
        pr, _ = scipy.stats.pearsonr(scores_epoch, truth_epoch)

        print("Mean Absolute Error: %.3f, Root Mean Squared Error %.3f, Pearson: %.3f" % (rmse, mae, pr))
    return rmse, mae, pr


def train_part(
    model,
    dataloader,
    optimizer,
    scheduler,
    val_loader,
    device,
    epochs=1,
    max_grad_norm=1.0,
    print_every=75,
    loss_function=F.mse_loss,
    return_metrics=True,
    val_every=None,
    return_losses=False,
    preprocessor: QuantileTransformer = None,
):
    """
    Train a BERTHoven model
    model: Instance of BERThoven model
    dataloader: pytorch Dataloader on which the model should be trained
    optimizer: pytorch Optimizer
    scheduler: pytorch scheduler
    val_loader: pytorch Dataloader on which the model accuracy is checked
    device: GPU or CPU
    epoch: number of epochs
    max_grad_norm: Clipping
    print_every: Controls the verbosity of training
    loss_function: Loss function used to backpropagate the gradients
    return_metrics: if True, returns the accuracy
    val_every: Controls the verobosity of accuracy
    return_losses: if True, returns the losses
    preprocessor: sklearn.preprocessing method, preprocess the scores according
    to the methods passed as input
    """

    avg_loss = None
    avg_val_loss = None
    momentum = 0.05

    t_losses = []
    v_losses = []

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Iterations per epoch:{len(dataloader)}")
        time.sleep(0.1)
        for t, (x1, x1_mask, x2, x2_mask, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            model.train()  # put model to training mode
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)

            scores = model.forward((x1, x1_mask), (x2, x2_mask))

            loss = loss_function(scores, y)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Effectively doubling the batch size
            # if t%2 ==0:
            #   torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            #   optimizer.step()
            #   optimizer.zero_grad()

            scheduler.step()
            l = loss.item()
            t_losses.append(l)
            if avg_loss is None:
                avg_loss = l
            else:
                avg_loss = l * momentum + avg_loss * (1 - momentum)

            if val_every is not None and t % val_every == 0:
                with torch.no_grad():
                    (x1, x1_mask, x2, x2_mask, y) = next(iter(val_loader))
                    x1 = x1.to(device=device, dtype=torch.long)
                    x1_mask = x1_mask.to(device=device, dtype=torch.long)
                    x2 = x2.to(device=device, dtype=torch.long)
                    x2_mask = x2_mask.to(device=device, dtype=torch.long)
                    y = y.to(device=device, dtype=torch.float32)
                    scores = model.forward((x1, x1_mask), (x2, x2_mask))
                    l_val = loss_function(scores, y).item()
                    v_losses.append(l_val)
                    if avg_val_loss is None:
                        avg_val_loss = l_val
                    else:
                        avg_val_loss = l_val * momentum + avg_val_loss * (1 - momentum)

            if t % print_every == 0:
                print()
                if avg_val_loss is not None:
                    print(
                        "Epoch: %d,\tIteration %d,\tMoving avg loss = %.4f\tval loss = %.4f"
                        % (e, t, avg_loss, avg_val_loss),
                        end="\t",
                    )
                else:
                    print("Epoch: %d,\tIteration %d,\tMoving avg loss = %.4f" % (e, t, avg_loss), end="\t")
            # print(".", end="")
        print()
        print("Checking accuracy on dev:")
        check_accuracy(val_loader, model, device=device, preprocessor=preprocessor)
        # print("Saving the model.")
        # torch.save(model.state_dict(), 'nlp_model.pt')
    if return_metrics:
        return check_accuracy(val_loader, model, device=device, preprocessor=preprocessor)
    if return_losses:
        px, py = smoothing(t_losses, 30)
        plt.plot(epochs * px, py, label="Training loss")
        px, py = smoothing(v_losses, 20)
        plt.plot(epochs * px, py, label="Validation loss")
        plt.legend()
        plt.show()
        return t_losses, v_losses


def get_test_labels(loader, model, device, preprocessor=None):
    """
    Returns the score given by a trained model on a test set.
    loader: pytorch Dataloader on which the model should predict the scores
    model: BERThoven model
    device: GPU or CPU
    preprocessor: sklearn.preprocessing method. If the model is trained with 
    preprocessing, the same preprocessor should be added

    """
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    all_scores = []
    with torch.no_grad():
        for x1, x1_mask, x2, x2_mask in loader:
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            scores = model.forward((x1, x1_mask), (x2, x2_mask))
            all_scores += [i.item() for i in scores]
    if preprocessor is not None:
        all_scores = np.array(all_scores).reshape(-1, 1)
        all_scores = preprocessor.inverse_transform(all_scores).reshape(-1)
    return all_scores


def writeScores(scores):
    """
    Write the scores to a file
    scores: numpy array, scores to be saved
    """
    fn = "predictions.txt"
    print("")
    with open(fn, "w") as output_file:
        for idx, x in enumerate(scores):
            output_file.write(f"{x}\n")
