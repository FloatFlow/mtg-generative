import argparse
from model.minigan import miniGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for miniGAN')

    # general parameters
    parser.add_argument('--train', type=bool, default=True, help='Whether to train.')
    parser.add_argument('--training_dir', type=str, default='Dataset/images', help='')
    parser.add_argument('--validation_dir', type=str, default='Training/validation_images', help='')
    parser.add_argument('--testing_dir', type=str, default='Testing/testing_images', help='')
    parser.add_argument('--checkpoint_dir', type=str, default='Training/model_saves', help='')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='Load weights from check point')
    parser.add_argument('--g_weights', type=str, default='Training/model_saves/minigan_generator_weights_65_0.162.h5', help='Load weights from check point')
    parser.add_argument('--d_weights', type=str, default='Training/model_saves/minigan_discriminator_weights_65_4.012.h5', help='Load weights from check point')
    parser.add_argument('--epochs', type=int, default=10000, help='')

    # model parameters
    parser.add_argument('--img_dim_x', type=int, default=256, help='')
    parser.add_argument('--img_dim_y', type=int, default=256, help='')
    parser.add_argument('--img_depth', type=int, default=3, help='')
    parser.add_argument('--z_len', type=int, default=128, help='')
    parser.add_argument('--class_embeddings', type=str, default='biggan', help="can be either 'biggan', 'onehot', or 'vanilla'")
    parser.add_argument('--n_classes', type=int, default=5, help='')
    parser.add_argument('--n_noise_samples', type=int, default=16, help='')
    parser.add_argument('--resblock_up_squeeze', type=int, default=3, help='')
    parser.add_argument('--resblock_down_squeeze', type=int, default=3, help='')
    parser.add_argument('--g_lr', type=float, default=4e-4, help='')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='')
    parser.add_argument('--g_d_update_ratio', type=int, default=1, help='')
    parser.add_argument('--d_g_update_ratio', type=int, default=1, help='')
    parser.add_argument('--save_freq', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=28, help='')
    parser.add_argument('--normalization', type=str, default='adain', help='either "batchnorm", "instancenorm", or "pixelnorm"')
    parser.add_argument('--kernel_init', type=str, default='ortho', help='either "norm" or "ortho"')
    parser.add_argument('--feat_matching', type=str, default='False')

    return parser.parse_args()

def main():
    args = parse_args()

    minigan = miniGAN(img_dim_x=args.img_dim_x,
                      img_dim_y=args.img_dim_y,
                      img_depth=args.img_depth,
                      z_len=args.z_len,
                      n_noise_samples=args.n_noise_samples,
                      resblock_up_squeeze=args.resblock_up_squeeze,
                      resblock_down_squeeze=args.resblock_down_squeeze,
                      g_lr=args.g_lr,
                      d_lr=args.d_lr,
                      g_d_update_ratio=args.g_d_update_ratio,
                      d_g_update_ratio=args.d_g_update_ratio,
                      save_freq=args.save_freq,
                      training_dir=args.training_dir,
                      validation_dir=args.validation_dir,
                      checkpoint_dir=args.checkpoint_dir,
                      testing_dir=args.testing_dir,
                      batch_size=args.batch_size,
                      conditional=args.class_embeddings,
                      n_classes=args.n_classes,
                      normalization=args.normalization,
                      kernel_init=args.kernel_init,
                      feat_matching=args.feat_matching)
    minigan.build_deep_generator()
    #minigan.build_dummy_generator()
    minigan.build_deep_discriminator()
    #minigan.build_dummy_discriminator()
    minigan.build_model()

    if args.train:
        if args.load_checkpoint:
            minigan.load_model_weights(args.g_weights, args.d_weights)
            print('Model Checkpoint Loaded...')
        minigan.train(args.epochs)

    else:
        minigan.predict_noise_testing(args.class_testing_labels,
                                      args.testing_dir)

    

if __name__ == '__main__':
    main()
