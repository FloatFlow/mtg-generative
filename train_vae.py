import argparse
import keras.backend as K
from model.vqvae import VQVAE
from model.vqvae2 import VQVAE2
import psutil

N_CPU = psutil.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for VQ-VAE-2')

    # general directory allocations
    parser.add_argument(
        '--train',
        type=bool,
        default=True
        )
    parser.add_argument(
        '--train_target',
        type=str,
        default='pixelcnn',
        help='can either be "pixelcnn" or "autoencoder"'
        )
    parser.add_argument(
        '--training_dir',
        type=str,
        #default='agglomerated_images'
        default='mtg_images'
        )
    parser.add_argument(
        '--validation_dir',
        type=str,
        default='logging/validation_images'
        )
    parser.add_argument(
        '--testing_dir',
        type=str,
        default='logging/testing_images'
        )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='logging/model_saves'
        )

    # load previous weights
    parser.add_argument(
        '--load_autoencoder_checkpoint',
        type=bool,
        default=True)
    parser.add_argument(
        '--encoder_weights',
        type=str,
        default='logging/model_saves/vqvae_encoder_weights_60_0.012.h5'
        )
    parser.add_argument(
        '--decoder_weights',
        type=str,
        default='logging/model_saves/vqvae_decoder_weights_60_0.012.h5'
        )
    parser.add_argument(
        '--load_pixelcnn_checkpoint',
        type=bool,
        default=False)
    parser.add_argument(
        '--pixelcnn_weights',
        type=str,
        default='logging/model_saves/vqvae_pixelcnn_weights_90_3.261.h5'
        )
    parser.add_argument(
        '--prior_sampler_weights',
        type=str,
        default='logging/model_saves/vqvae_pixelsampler_weights_90_3.261.h5'
        )

    

    # model parameters
    parser.add_argument(
        '--img_dim_x',
        type=int,
        default=256
        )
    parser.add_argument(
        '--img_dim_y',
        type=int,
        default=256
        )
    parser.add_argument(
        '--img_depth',
        type=int,
        default=3
        )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4
        )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=10
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
        )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000
        )
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=N_CPU
        )

    return parser.parse_args()

def main():
    args = parse_args()
    if args.train_target == 'pixelcnn':
        assert args.load_autoencoder_checkpoint

    vae = VQVAE2(
        img_dim_x=args.img_dim_x,
        img_dim_y=args.img_dim_y,
        img_depth=args.img_depth,
        lr=args.lr,
        save_freq=args.save_freq,
        training_dir=args.training_dir,
        validation_dir=args.validation_dir,
        checkpoint_dir=args.checkpoint_dir,
        testing_dir=args.testing_dir,
        batch_size=args.batch_size,
        n_cpu=args.n_cpu
        )

    if args.train:
        if args.load_autoencoder_checkpoint:
            vae.encoder.load_weights(args.encoder_weights, by_name=True)
            vae.decoder.load_weights(args.decoder_weights, by_name=True)
            print('Successfully loaded autoencoder checkpoints...')
        if args.train_target == 'autoencoder':
            vae.train(args.epochs)

        else:
            vae.build_pixelcnn()
            if args.load_pixelcnn_checkpoint:
                vae.pixelcnn.load_weights(args.pixelcnn_weights, by_name=True)
                vae.pixel_sampler.load_weights(args.prior_sampler_weights, by_name=True)
                print('Successfully loaded pixelcnn checkpoints...')
            vae.train_pixelcnn(args.epochs)
    else:
        vae.encoder.load_weights(args.encoder_weights, by_name=True)
        vae.decoder.load_weights(args.decoder_weights, by_name=True)
        vae.build_pixelcnn()
        vae.pixelcnn.load_weights(args.pixelcnn_weights, by_name=True)
        vae.pixelsampler.load_weights(args.prior_sampler_weights, by_name=True)
        print('Successfully loaded model checkpoints...')
        for i in range(10):
            vae.generate_samples(i)

if __name__ == '__main__':
    main()
