import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset
from dev_basics.utils import vid_io

# unet for imagen

unet1 = Unet(
    dim = 64,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 64,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)


imagen = Imagen(
    unets = (unet1, unet2),
    # text_encoder_name = 't5-large',
    condition_on_text = False,
    image_sizes = (64, 256),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# wrap imagen with the trainer class

trainer = ImagenTrainer(imagen,split_valid_from_train = True).cuda()

# mock images (get a lot of this) and text encodings from large T5

# text_embeds = torch.randn(4, 256, 768).cuda()
# images = torch.randn(4, 3, 256, 256).cuda()
# text_embeds = torch.randn(4, 256, 768).cuda()
# images = torch.randn(4, 3, 128, 128).cuda()
# text_embeds = torch.randn(64, 256, 1024).cuda()
# images = torch.randn(64, 3, 256, 256).cuda()


# feed images into imagen, training each unet in the cascade
fn = "/home/gauenk/Documents/data/coco/images/val2014"
dataset = Dataset(fn, image_size = 256)

trainer.add_train_dataset(dataset, batch_size = 12)

# working training loop

for i in range(200000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    print(f'loss: {loss}')

    if not (i % 50) and (i>0):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        print(f'valid loss: {valid_loss}')

    if not (i % 100) and trainer.is_main and (i>0): # is_main makes sure this can run in distributed
        images = trainer.sample(batch_size = 1, return_pil_images = True,
                                stop_at_unet_number = 1) # returns List[Image]
        images[0].save(f'./sample-{i // 100}.png')
trainer.save('./checkpoint.pt')

images = trainer.sample(batch_size = 1, return_pil_images = True)
images = np.stack([np.array(img) for img in images])
# images = imagen.sample(texts = [
#     'a whale breaching from afar',
#     'young girl blowing out candles on her birthday cake',
#     'fireworks with blue and green sparkles'
# ], cond_scale = 3.)

# images.shape # (3, 3, 256, 256)

vid_io.save_video(images,"output","example")
