import torch

def inference(model, testloader, device):
    """
    Perform inference on the test set.
    
    Parameters:
        model (nn.Module): Trained CVAE model.
        testloader (DataLoader): Test data loader.
        device (torch.device): Device (CPU or CUDA).
    
    Returns:
        tuple: (reconstructed spectrograms, genres input, original spectrograms)
    """
    model.eval()
    with torch.no_grad():
        data, genres_input, ori_data = next(iter(testloader))
        data = data.to(device)
        genres_input = genres_input.to(device)
        recon, _, _ = model(data, genres_input)
        return recon, genres_input, ori_data
