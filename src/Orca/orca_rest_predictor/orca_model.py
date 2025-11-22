# orca_model.py
import sys, torch, warnings
from selene_sdk.sequences import Genome

ORCA_PATH = "/orca/"
USE_CUDA = torch.cuda.is_available()
sys.path.append(ORCA_PATH)

import orca_predict
orca_predict.load_resources(models=["1M"], use_cuda=USE_CUDA)
from orca_predict import h1esc_1m  # or hff_1m

def orca_prediction(sequences_dict):
    """
    sequences_dict: {seq_id: DNA_string}
    returns: {seq_id: np.ndarray (250 x 250)}
    """
    predictions = {}
    for seq_id, seq in sequences_dict.items():
        sequence_encoded = Genome.sequence_to_encoding(seq)[None, :, :]
        model = h1esc_1m
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pred = model(torch.FloatTensor(sequence_encoded).transpose(1, 2))
        predictions[seq_id] = pred[0][0].cpu().detach().numpy()
    return predictions