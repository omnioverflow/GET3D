import click
import os

# python train_3d.py
#  --outdir=save_inference_results/shapenet_car
#    --gpus=1 --batch=4
#     --gamma=40 
#     --data_camera_mode shapenet_car
#       --dmtet_scale 1.0
#         --use_shapenet_split 1
#           --one_3d_generator 1
#             --fp32 0
#              --inference_vis 1
#               --resume_pretrain MODEL_PATH

# ==============================================================================
gen_tex_mesh = False
model_type = "shapenet_car"
# ==============================================================================

command = "train_3d.py"
command += f" --outdir=save_inference_results/{model_type}"
command += " --gpus=1 --batch=4"
command += " --gamma=40"
command += f" --data_camera_mode {model_type}"
command += " --dmtet_scale 1.0"
command += " --use_shapenet_split 1"
command += " --one_3d_generator 1"
command += " --fp32 0"
command += " --inference_vis 1"
if gen_tex_mesh:
    command += " --inference_to_generate_textured_mesh 1"
command += f" --resume_pretrain get3d_release/{model_type}.pt"

@click.command()
@click.option("--model_type", default="shapenet_car", help="Name of a model e.g. shapenet_car, shapenet_chair, shapenet_motorbike ")
@click.option("--gen_tex_mesh", default=False, help="Set to True if texture must be generated")
def main(model_type, gen_tex_mesh):
    print(command)

if __name__ == "__main__":
    main()
