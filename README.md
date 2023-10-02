# Requirements

Please make sure your GPU(>=40G)

ubuntu20.04/other Linux System

python3.10

# Training

1. clone this repo

   `git clone https://github.com/armada-ai/MyLAVIS.git`

2. enter the repo directory

   `cd MyLAVIS`

3. installation

   `pip install salesforce-lavis`
   `pip install -e .`

4. data preparation

   1. unzip the database to the specific directory

      `sudo unzip sample20230919.zip -d /content`

   2. gt annotation preparation

      `sudo mkdir -p /export/home/.cache/lavis/coco_gt`

      `sudo chmod 777 -R /export/home/.cache/lavis/coco_gt`

      `python gen_coco_format_script.py`

5. training

   `python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_sample20230919_ft.yaml`

6. testing

   `python demo.py`

   Please modify `checkpoint` to your own checkpoint path.

# Reference

[https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)

