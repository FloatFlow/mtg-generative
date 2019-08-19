import argparse
import keras.backend as K
from model.vqvae import VQVAE
import psutil

N_CPU = psutil.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for VQ-VAE-2')

    # general parameters
    parser.add_argument('--train',
                        type=bool,
                        default=True)
    parser.add_argument('--training_dir',
                        type=str,
                        default='data/mtg_images')
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
                        default='logging/model_saves/stylegan_hinge_generator_weights_18_-0.002.h5')
    parser.add_argument('--d_weights',
                        type=str,
                        default='logging/model_saves/stylegan_hinge_discriminator_weights_18_4.003.h5')
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
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4)
    parser.add_argument('--save_freq',
                        type=int,
                        default=2)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)

    return parser.parse_args()

def main():
    args = parse_args()
    K.set_floatx('float32')


    vae = VQVAE(img_dim_x=args.img_dim_x,
                img_dim_y=args.img_dim_y,
                img_depth=args.img_depth,
                lr=args.lr,
                save_freq=args.save_freq,
                training_dir=args.training_dir,
                validation_dir=args.validation_dir,
                checkpoint_dir=args.checkpoint_dir,
                testing_dir=args.testing_dir,
                batch_size=args.batch_size,
                n_cpu=args.n_cpu)

    vae.build_model()

    if args.train:
        if args.load_checkpoint:
            vae.load_model_weights(args.decoder_weights, args.encoder_weights)
            print('Model Checkpoint Loaded...')

        vae.train(args.epochs)

    else:
        vae.predict_noise_testing(args.class_testing_labels,
                                      args.testing_dir)


if __name__ == '__main__':
    main()
