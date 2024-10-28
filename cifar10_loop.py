import numpy as np

from rich.console import Console
from rich.table import Table

from cifar10_simulations import main


GPU_ID = 0
HIDDEN_LAYERS = [3, 4, 5, 6]
HIDDEN_FEATURES = [22, 18, 16, 14]
N_IMAGES = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
CLASS_NAMES = [str(i) for i in range(10)]


if __name__ == '__main__':
    console = Console()

    table = Table(title="Mixture Model Results")
    table.add_column("Class Name", style="purple")
    table.add_column("Images", style="red")
    table.add_column("Network structure", style="cyan")
    table.add_column("Average PSNR Half", style="green")
    table.add_column("Std. Dev. PSNR Half", style="yellow")
    table.add_column("Training Time", style="blue")

    for class_name in CLASS_NAMES:
        for n_images in N_IMAGES:
            psnrs_avg = np.zeros(len(HIDDEN_LAYERS))
            psnrs_std = np.zeros(len(HIDDEN_LAYERS))
            for i in range(len(HIDDEN_LAYERS)):
                average_psnr_half, std_psnr_half, training_time = main(
                    hidden_layers=HIDDEN_LAYERS[i],
                    hidden_features=HIDDEN_FEATURES[i],
                    gpu_id=GPU_ID,
                    n_images=n_images,
                    class_name=class_name
                )
                psnrs_avg[i] = average_psnr_half
                psnrs_std[i] = std_psnr_half

            best_idx = np.argmax(psnrs_avg)
            average_psnr_half = psnrs_avg[best_idx]
            std_psnr_half = psnrs_std[best_idx]

            table.add_row(
                class_name,
                f"{n_images} images",
                f"{HIDDEN_LAYERS[best_idx]} x {HIDDEN_FEATURES[best_idx]}",
                f"{average_psnr_half:.2f}",
                f"{std_psnr_half:.2f}",
                f"{training_time:.2f}"
            )

    console.print(table)
