from rich.console import Console
from rich.table import Table

from kodak_simulations import main

### BPP = 0.4
# GPU_ID = 1
# HIDDEN_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9, 10]#, 12, 14, 16]
# HIDDEN_FEATURES = [294, 208, 170, 148, 132, 120, 112, 104, 98]#, 89, 82, 76]
# N_IMAGES = 6

### BPP = 1.0
# GPU_ID = 0
# HIDDEN_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
# HIDDEN_FEATURES = [466, 330, 270, 234, 210, 192, 176, 166, 156, 142, 130, 120]
# N_IMAGES = 6

# GPU_ID = 0
# HIDDEN_LAYERS = [8, 8]
# HIDDEN_FEATURES = [158, 194]
# N_IMAGES = 6

# GPU_ID = 0
# HIDDEN_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# HIDDEN_FEATURES = [206, 146, 120, 104, 92, 84, 78, 74, 70]
# N_IMAGES = 6

# GPU_ID = 1
# HIDDEN_LAYERS = [8]
# HIDDEN_FEATURES = [112]
# N_IMAGES = [2, 3, 4, 12]

# GPU_ID = 1
# HIDDEN_LAYERS = [8, 8]
# HIDDEN_FEATURES = [96, 56]
# N_IMAGES = 6

GPU_ID = 1
HIDDEN_LAYERS = 6 * [8]
HIDDEN_FEATURES = [158, 136, 112, 96, 78, 56]
N_IMAGES = [6]


SORTED_FREQ = [11, 19, 22, 8, 2, 9, 6, 3, 14, 15, 21, 1, 5, 18, 20, 16, 10, 13, 0, 23, 7, 4, 17, 12]

if __name__ == '__main__':
    console = Console()

    table = Table(title="Mixture Model Results")
    table.add_column("Images", style="red")
    table.add_column("Network structure", style="cyan")
    table.add_column("Average PSNR Float", style="magenta")
    table.add_column("Average PSNR Half", style="green")
    table.add_column("Training Time", style="blue")

    total_images = 24
    for n_images in N_IMAGES:
        image_groups = [SORTED_FREQ[i:total_images:total_images//n_images] for i in range(0, total_images // n_images)]

        for i in range(len(HIDDEN_LAYERS)):
            avg_psnr_floats = []
            avg_psnr_halfs = []
            training_times = []
            for img_idx in image_groups:
                average_psnr_float, average_psnr_half, training_time, _ = main(
                    hidden_layers=HIDDEN_LAYERS[i],
                    hidden_features=HIDDEN_FEATURES[i],
                    gpu_id=GPU_ID,
                    image_idxs=img_idx
                )

                avg_psnr_floats.append(average_psnr_float)
                avg_psnr_halfs.append(average_psnr_half)
                training_times.append(training_time)

            average_psnr_float = sum(avg_psnr_floats) / len(avg_psnr_floats)
            average_psnr_half = sum(avg_psnr_halfs) / len(avg_psnr_halfs)
            training_time = sum(training_times) / len(training_times)

            table.add_row(
                f"{n_images} images",
                f"{HIDDEN_LAYERS[i]} x {HIDDEN_FEATURES[i]}",
                f"{average_psnr_float:.2f}",
                f"{average_psnr_half:.2f}",
                f"{training_time:.2f} s"
            )

    console.print(table)
