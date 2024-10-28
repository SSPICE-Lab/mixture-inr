import os

from rich.console import Console
from rich.table import Table

from imagenet_simulations import main

DATAPATH = "/data/shared/datasets/images/imagenet/resized"
CLASSES = os.listdir(DATAPATH)
CLASSES.sort()

GPU_ID = 1
HIDDEN_LAYERS = 7
HIDDEN_FEATURES = 49
N_IMAGES = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
CLASS_NAMES = CLASSES[20:100]


if __name__ == '__main__':
    console = Console()

    table = Table(title="Mixture Model Results")
    table.add_column("Class Name", style="purple")
    table.add_column("Images", style="red")
    table.add_column("Network structure", style="cyan")
    table.add_column("Average PSNR Float", style="magenta")
    table.add_column("Average PSNR Half", style="green")
    table.add_column("Training Time", style="blue")

    for class_name in CLASS_NAMES:
        for n_images in N_IMAGES:
            average_psnr_float, average_psnr_half, training_time = main(
                hidden_layers=HIDDEN_LAYERS,
                hidden_features=HIDDEN_FEATURES,
                gpu_id=GPU_ID,
                n_images=n_images,
                class_name=class_name
            )

            if average_psnr_float is not None:
                table.add_row(
                    class_name,
                    f"{n_images} images",
                    f"{HIDDEN_LAYERS} x {HIDDEN_FEATURES}",
                    f"{average_psnr_float:.2f}",
                    f"{average_psnr_half:.2f}",
                    f"{training_time:.2f}"
                )
            else:
                break

    console.print(table)
