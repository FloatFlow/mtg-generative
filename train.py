import argparse
import keras.backend as K
#from model.stylegan import StyleGAN
#from model.minigan import MiniGAN
#from model.msgstylegan import MSGStyleGAN
#from model.styleaae import StyleAAE
from model.pixelaae import PixelAAE
import psutil

N_CPU = psutil.cpu_count()
print('Allocating {} Processes for data loading...'.format(N_CPU))

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for GAN')

    # general parameters
    parser.add_argument(
        '--train',
        type=bool,
        default=True
        )
    parser.add_argument(
        '--training_dir',
        type=str,
        default='images'
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
    parser.add_argument(
        '--load_checkpoint',
        type=bool,
        default=False
        )

    # train parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000
        )
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=8
        )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4
        )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=5
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
        )

    return parser.parse_args()

def main():
    args = parse_args()
    gan = PixelAAE(
        lr=args.lr,
        training_dir=args.training_dir,
        validation_dir=args.validation_dir,
        checkpoint_dir=args.checkpoint_dir,
        testing_dir=args.testing_dir,
        )

    if args.train:
        if args.load_checkpoint:
            epoch = 25
            loss = 0.119
            gan.encoder.load_weights(
                'logging/model_saves/styleaae_encoder_weights_{}_{}.h5'.format(epoch, loss),
                by_name=True
                )
            gan.decoder.load_weights(
                'logging/model_saves/styleaae_encoder_weights_{}_{}.h5'.format(epoch, loss),
                by_name=True
                )
            gan.style_discriminator.load_weights(
                'logging/model_saves/styleaae_sd_weights_{}_{}.h5'.format(epoch, loss),
                by_name=True
                )
            gan.color_discriminator.load_weights(
                'logging/model_saves/styleaae_cd_weights_{}_{}.h5'.format(epoch, loss),
                by_name=True
                )
            print('Success - Model Checkpoint Loaded...')
        gan.train(
            epochs=args.epochs,
            n_cpu=args.n_cpu,
            batch_size=args.batch_size,
            save_freq=args.save_freq
            )

    else:
        gan.predict_noise_testing(
            args.class_testing_labels,
            args.testing_dir
            )

if __name__ == '__main__':
    main()
