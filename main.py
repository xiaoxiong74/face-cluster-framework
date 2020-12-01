import argparse
from face_feature_extract import models
from face_cluster.face_cluster_by_infomap import cluster_main
from face_feature_extract.extract_feature import extract_fature
from tools.utils import Timer


parser = argparse.ArgumentParser(description='Face Cluster')
parser.add_argument('--is_cuda', default='True', type=str)

# feature-extract config
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--input-size', default=112, type=int)
parser.add_argument('--feature-dim', default=256, type=int)
parser.add_argument('--load-path', default='pretrain_models/res50_softmax.pth.tar', type=str)
parser.add_argument('--strict', dest='strict', action='store_true')
parser.add_argument('--output-path', default='bin/test.bin', type=str)

# cluster config
parser.add_argument('--input_picture_path', default='data/input_pictures/data_sample', type=str)
parser.add_argument('--output_picture_path', default='data/output_pictures/data_sample', type=str)
parser.add_argument('--knn_method', default='faiss-gpu', type=str)
parser.add_argument('--is_evaluate', default='False', type=str)
parser.add_argument('--k', default=80, type=int)
parser.add_argument('--min_sim', default=0.5, type=float)
parser.add_argument('--metrics', default=['pairwise', 'bcubed', 'nmi'], type=list)
parser.add_argument('--label_path', default='data/tmp/test.meta', type=str)
parser.add_argument('--save_result', default='False', type=str)

if __name__ == '__main__':
    with Timer('All Steps'):
        global args
        args = parser.parse_args()
        label_path = None
        pred_label_path = None
        print('=> Use cuda ?: {}'.format(args.is_cuda))
        with Timer('Extract Feature'):
            extract_features = extract_fature(args)
        if eval(args.is_evaluate):
            args.label_path = 'data/tmp/test.meta'
        if not eval(args.is_cuda):
            args.knn_method = 'faiss-cpu'
        with Timer('Face Cluster'):
            cluster_main(args, extract_features)
        print("=> Face cluster done! The cluster results have been saved in {}".format(args.output_picture_path))
