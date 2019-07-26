import argparse
from model.minigan import MiniGAN
from model.stylegan import StyleGAN
import psutil

N_CPU = psutil.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for miniGAN')

    # general parameters
    parser.add_argument('--model_type',
                        type=str,
                        default='stylegan')
    parser.add_argument('--train',
                        type=bool,
                        default=True)
    parser.add_argument('--training_dir',
                        type=str,
                        default='data/images')
    parser.add_argument('--validation_dir',
                        type=str,
                        default='logging/validation_images')
    parser.add_argument('--testing_dir',
                        type=str,
                        default='logging/testing_images')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='logging/model_saves')
    parser.add_argument('--load_checkpoint',
                        type=bool,
                        default=False)
    parser.add_argument('--g_weights',
                        type=str,
                        default='logging/model_saves/minigan_generator_weights_15_-0.116.h5')
    parser.add_argument('--d_weights',
                        type=str,
                        default='logging/model_saves/minigan_discriminator_weights_15_3.923.h5')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000)
    parser.add_argument('--n_cpu',
                        type=int,
                        default=N_CPU)

    # model parameters
    parser.add_argument('--img_dim_x',
                        type=int,
                        default=256)
    parser.add_argument('--img_dim_y',
                        type=int,
                        default=256)
    parser.add_argument('--img_depth',
                        type=int,
                        default=3)
    parser.add_argument('--z_len',
                        type=int,
                        default=256)
    parser.add_argument('--n_classes',
                        type=int,
                        default=5)
    parser.add_argument('--g_lr',
                        type=float,
                        default=4e-4)
    parser.add_argument('--d_lr',
                        type=float,
                        default=5e-5)
    parser.add_argument('--save_freq',
                        type=int,
                        default=5)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.model_type == 'minigan':
        gan = MiniGAN(img_dim_x=args.img_dim_x,
                      img_dim_y=args.img_dim_y,
                      img_depth=args.img_depth,
                      z_len=args.z_len,
                      g_lr=args.g_lr,
                      d_lr=args.d_lr,
                      save_freq=args.save_freq,
                      training_dir=args.training_dir,
                      validation_dir=args.validation_dir,
                      checkpoint_dir=args.checkpoint_dir,
                      testing_dir=args.testing_dir,
                      batch_size=args.batch_size,
                      n_classes=args.n_classes,
                      n_cpu=args.n_cpu)
    elif args.model_type == 'stylegan':
        gan = StyleGAN(img_dim_x=args.img_dim_x,
                       img_dim_y=args.img_dim_y,
                       img_depth=args.img_depth,
                       z_len=args.z_len,
                       g_lr=args.g_lr,
                       d_lr=args.d_lr,
                       save_freq=args.save_freq,
                       training_dir=args.training_dir,
                       validation_dir=args.validation_dir,
                       checkpoint_dir=args.checkpoint_dir,
                       testing_dir=args.testing_dir,
                       batch_size=args.batch_size,
                       n_classes=args.n_classes,
                       n_cpu=args.n_cpu)
    else:
        raise ValueError('Invalid model type: {}'.format(args.model_type))

    gan.build_generator()
    gan.build_discriminator()
    gan.build_model()

    if args.train:
        if args.load_checkpoint:
            gan.load_model_weights(args.g_weights, args.d_weights)
            print('Model Checkpoint Loaded...')

        gan.train(args.epochs)

    else:
        gan.predict_noise_testing(args.class_testing_labels,
                                      args.testing_dir)


if __name__ == '__main__':
    main()
