import argparse
import keras.backend as K
from model.stylegan2 import StyleGAN2
from model.evolstylegan import EvolStyleGAN
import psutil
import pickle

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
        default=True
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
        default=20
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
        )

    return parser.parse_args()

def main():
    args = parse_args()
    gan = StyleGAN2(
        lr=args.lr,
        training_dir=args.training_dir,
        validation_dir=args.validation_dir,
        checkpoint_dir=args.checkpoint_dir,
        testing_dir=args.testing_dir,
        img_depth=3,
        img_height=256,
        img_width=256,
        )
    if args.load_checkpoint:
        gan.generator.load_weights(
            'logging/model_saves/stylegan2_generator_weights_150_0.048.h5',
            by_name=True
            )
        gan.discriminator.load_weights(
            'logging/model_saves/stylegan2_discriminator_weights_150_0.426.h5',
            by_name=True
            )
        #gan.load_quality_estimator('logging/model_saves/koncept/bsz64_i1[224,224,3]_lMSE_o1[1]_best_weights.h5')
        print('Success - Model Checkpoints Loaded...')
        gan.generator_model._make_train_function()
        with open('logging/model_saves/stylegan2_generator_optimizer_state_150_0.048.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        gan.generator_model.optimizer.set_weights(weight_values)
        gan.discriminator_model._make_train_function()
        with open('logging/model_saves/stylegan2_discriminator_optimizer_state_150_0.426.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        gan.discriminator_model.optimizer.set_weights(weight_values)
        print('Success - Model Optimizer States Loaded...')

    if args.train:
        gan.train(
            epochs=args.epochs,
            n_cpu=args.n_cpu,
            batch_size=args.batch_size,
            save_freq=args.save_freq
            )

    else:
        gan.generate_samples(
            savedir='logging/testing_images',
            n_samples=8196,
            batch_size=args.batch_size,
            z_var=0.75
            )

if __name__ == '__main__':
    main()
