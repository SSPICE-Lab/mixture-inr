import os
import time

import cv2
import numpy as np
import torch
import torch.utils.data

import inr
from mixtures.mixture_network import MixtureNetwork


DATAPATH = "/data/shared/datasets/images/imagenet/resized"

CLASS_NAME = "n01440764"
N_IMAGES = 96

EPOCHS = 5000
BATCH_SIZE = 400000
HIDDEN_LAYERS = 8
HIDDEN_FEATURES = 49

SEED = 0
GPU_ID = 1

T0 = 500
TMUL = 1
ETA_MIN = 5e-6

def main(**kwargs):
    datapath = kwargs.get("datapath", DATAPATH)
    class_name = kwargs.get("class_name", CLASS_NAME)
    n_images = kwargs.get("n_images", N_IMAGES)
    epochs = kwargs.get("epochs", EPOCHS)
    batch_size = kwargs.get("batch_size", BATCH_SIZE)
    seed = kwargs.get("seed", SEED)
    hidden_layers = kwargs.get("hidden_layers", HIDDEN_LAYERS)
    hidden_features = kwargs.get("hidden_features", HIDDEN_FEATURES)
    gpu_id = kwargs.get("gpu_id", GPU_ID)
    t0 = kwargs.get("t0", T0)
    tmul = kwargs.get("tmul", TMUL)
    eta_min = kwargs.get("eta_min", ETA_MIN)

    def compute_function(weight_list):
        ret_list = []
        for i in range(n_images):
            ret_list.append(weight_list[0] * (1 - i / (n_images-1)) + weight_list[1] * (i / (n_images-1)))

        return ret_list

    torch.manual_seed(seed)

    image_list = os.listdir(os.path.join(datapath, class_name))
    image_list.sort()
    image_list = image_list[:n_images]

    run_name = f"imagenet_{class_name}_{n_images}_{epochs}_{hidden_layers}_{hidden_features}"

    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    image_data_list = [inr.data.image.ImageData(
        os.path.join(datapath, class_name, image)).pixels for image in image_list]
    image_data = torch.stack(image_data_list, dim=1)
    coords = inr.data.CoordGrid(2, (256, 256)).coord_grid

    # Create dataset
    dataset = torch.utils.data.TensorDataset(coords, image_data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available()
    )

    # Create the neural network
    net = MixtureNetwork(
        input_features=2,
        output_features=3,
        hidden_features=[hidden_features] * hidden_layers,
        n_weights=2,
        n_images=n_images,
        compute_function=compute_function,
        activation=inr.layer.Activation.SINE,
        outermost_activation=inr.layer.Activation.LINEAR,
        first_scale_factor=30,
        hidden_scale_factor=30,
        bias=True
    ).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=tmul, eta_min=eta_min
    )

    # Train the network
    save_path = f"test_models/{run_name}.pt"
    net.train()
    start_time = time.time()
    loss_history = net.fit(
        dataloader,
        epochs=epochs,
        optimizer=optimizer,
        verbose=True,
        save_path=save_path,
        device=device,
        scheduler=lr_scheduler,
        scheduler_args='epoch'
    )
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} s")

    # Save the loss history
    np.save(f"test_models/{run_name}_loss.npy", loss_history)

    ## Get the average PSNR for the float model
    if not os.path.exists(save_path):
        return None, None, None
    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()

    # Generate the test coordinates
    test_coords = inr.data.CoordGrid(2, (256, 256)).coord_grid
    test_dataloader = torch.utils.data.DataLoader(test_coords, batch_size=batch_size//2, shuffle=False)

    # Generate the test images
    test_images = net.generate(test_dataloader, device=device)

    # Save the test images
    os.makedirs(f"test_images/{run_name}", exist_ok=True)
    for i in range(n_images):
        inr.data.utils.save_image(
            test_images[:, i, :],
            f"test_images/{run_name}/image_{i}.png",
            image_size=(256, 256)
        )

    # Compute the average PSNR
    psnrs = np.zeros(n_images)
    for i in range(n_images):
        image = cv2.imread(f"test_images/{run_name}/image_{i}.png")
        original = cv2.imread(os.path.join(datapath, class_name, image_list[i]))
        psnrs[i] = cv2.PSNR(image, original)

    np.save(f"test_images/{run_name}/float_psnrs.npy", psnrs)
    average_psnr_float = psnrs.mean()
    print(f"Average PSNR: {psnrs.mean():.2f} dB")

    # Convert the model to half precision
    net.load_state_dict(torch.load(save_path, map_location=device))
    net.half()
    torch.save(net.state_dict(), f"test_models/{run_name}_half.pt")
    net.load_state_dict(torch.load(f"test_models/{run_name}_half.pt"))
    net.float()

    # Generate the test images
    test_images = net.generate(test_dataloader, device=device)

    # Save the test images
    for i in range(n_images):
        inr.data.utils.save_image(
            test_images[:, i, :],
            f"test_images/{run_name}/image_{i}_half.png",
            image_size=(256, 256)
        )

    # Compute the average PSNR
    psnrs = np.zeros(n_images)
    for i in range(n_images):
        image = cv2.imread(f"test_images/{run_name}/image_{i}_half.png")
        original = cv2.imread(os.path.join(datapath, class_name, image_list[i]))
        psnrs[i] = cv2.PSNR(image, original)

    np.save(f"test_images/{run_name}/half_psnrs.npy", psnrs)
    average_psnr_half = psnrs.mean()
    print(f"Average PSNR Half: {psnrs.mean():.2f} dB")

    return average_psnr_float, average_psnr_half, training_time

if __name__ == "__main__":
    main()
