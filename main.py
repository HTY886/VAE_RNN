import argparse
import tensorflow as tf
from model import vrnn

def parse():
    parser = argparse.ArgumentParser(description="variational autoencoder")
    parser.add_argument('-model_dir','--model_dir',default='model_dir',help='output model dir')
    parser.add_argument('-model_path','--model_path',help='latest model path')
    parser.add_argument('-batch_size','--batch_size',default=48,type=int,help='batch size')
    parser.add_argument('-latent_dim','--latent_dim',default=500,type=int,help='laten size')
    parser.add_argument('-data_dir','--data_dir',default='data_dir',help='data dir')
    parser.add_argument('-saving_step','--saving_step',default=1000,type=int,help='saving step')
    parser.add_argument('-num_steps','--num_steps',default=60000,type=int,help='number of steps')
    parser.add_argument('-sequence_length','--sequence_length',default=30,type=int,help='sentence length')
    parser.add_argument('-load','--load',action='store_true',help='whether load')
    parser.add_argument('-train',action='store_true',help='whether train')
    parser.add_argument('-stdin_test',action='store_true',help='whether stdin test')
    args = parser.parse_args()
    return args

def run(args):
    sess = tf.Session()
    model = vrnn(args,sess)
    if args.train:
        model.train()
    if args.stdin_test:
        model.stdin_test()

if __name__ == '__main__':
    args = parse()
    run(args)

