import argparse
import keras.backend as K
#from model.stylegan import StyleGAN
#from model.minigan import MiniGAN
#from model.msgstylegan import MSGStyleGAN
from model.stylegan2_pl import StyleGAN2
import psutil

N_CPU = psutil.cpu_count()

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
    parser.add_argument(
        '--load_checkpoint',
        type=bool,
        default=False
        )
    parser.add_argument(
        '--g_weights',
        type=str,
        default='logging/model_saves/stylegan2_generator_weights_30_1.576.h5'
        )
    parser.add_argument(
        '--d_weights',
        type=str,
        default='logging/model_saves/stylegan2_discriminator_weights_30_1.150.h5'
        )
    parser.add_argument(
        '--m_weights',
        type=str,
        default='logging/model_saves/stylegan2_discriminator_weights_2_0.981.h5'
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
        '--z_len',
            type=int,
            default=256
            )
    parser.add_argument(
        '--n_classes',
            type=int,
            default=5
            )
    parser.add_argument(
        '--g_lr',
            type=float,
            default=1e-4
            )
    parser.add_argument(
        '--d_lr',
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
        default=16
        )

    return parser.parse_args()

def main():
    args = parse_args()
    #gan = MiniGAN(
    gan = StyleGAN2(
        img_dim_x=args.img_dim_x,
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
        n_cpu=args.n_cpu
        )

    if args.train:
        if args.load_checkpoint:
            gan.discriminator.load_weights(args.d_weights, by_name=True)
            gan.generator.load_weights(args.g_weights, by_name=True)
            gan.mapper.load_weights(args.m_weights, by_name=True)
            print('Success - Model Checkpoint Loaded...')
        gan.build_model()
        gan.train(args.epochs)

    else:
        gan.predict_noise_testing(
            args.class_testing_labels,
            args.testing_dir
            )

if __name__ == '__main__':
    main()
