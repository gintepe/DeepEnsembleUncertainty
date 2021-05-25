CORRUPTIONS = ['brightness', 'defocus_blur', 'elastic_transform', 
                'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 
                'glass_blur', 'jpeg_compression', 'saturate', 
                'shot_noise', 'snow', 'spatter', 
                'speckle_noise', 'zoom_blur', 'motion_blur',
                'contrast', 'impulse_noise', 'pixelate']
CIFAR10_TEST_N = 10000
CIFAR10_SIZE = (32,32)
CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

DATA_DIR = '/scratch/gp491/data'
LOGGING_DIR = '/scratch/gp491/wandb'
CHECKPOINT_DIR = f'{LOGGING_DIR}/checkpoints'