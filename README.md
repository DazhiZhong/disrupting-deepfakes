# StarGAN Adversarial Attack
⚠️ This project is a forked version of [this repo](https://github.com/DazhiZhong/disrupting-deepfakes)! Please go there for the original code.

![examples stargen](README/examples%20stargen-0092376.png)

In our paper, we propose AI-FGSM, a method for transferable adversarial attack for a wide range of models. We experiment of three deep neural networks consisting of image translation and image classification networks. This is our second experiment, which is done on face swapping networks. See our other expeirments at  [this repo](https://github.com/DazhiZhong/deepfakes_faceswap) (faceswap) and [this repo](https://github.com/jasonliuuu/SI-AI-FGSM) (Inception models).



**StarGAN Dataset**

```
cd stargan
bash download.sh celeba
```

**StarGAN Models**

```
bash download.sh pretrained-celeba-128x128
```

**Attacking StarGAN Model**

```
cd stargan
python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_128/models' --result_dir='stargan_celeba_128/results_test' --test_iters 200000 --batch_size 1
```

Check the CelebA attributes [here](https://www.tensorflow.org/datasets/catalog/celeb_a)



To modify the code, you can alter the `LinfPGDAttack` class in `attacks.py` or alter the `test_attack` method in the `Solver` class in `solver.py`



Alternatively, use the jupyter notebook in the files, or try the colab notebook [here](https://colab.research.google.com/drive/1UCjgBTdXA4UGL4L3OQep8EGtId1B79mD?usp=sharing) for testing.
