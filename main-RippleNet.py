from src.RippleNet import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=40, help='embedding size')
    # parser.add_argument('--H', type=int, default=3, help='H')
    # parser.add_argument('--K_h', type=int, default=40, help='The size of ripple set')
    # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=50, help='embedding size')
    # parser.add_argument('--H', type=int, default=3, help='H')
    # parser.add_argument('--K_h', type=int, default=50, help='The size of ripple set')
    # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument("--device", type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=20, help='embedding size')
    # parser.add_argument('--H', type=int, default=1, help='H')
    # parser.add_argument('--K_h', type=int, default=50, help='The size of ripple set')
    # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
    #
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument("--device", type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=40, help='embedding size')
    parser.add_argument('--H', type=int, default=2, help='H')
    parser.add_argument('--K_h', type=int, default=20, help='The size of ripple set')
    parser.add_argument('--l1', type=float, default=1e-2, help='L1')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.936 	 train_acc: 0.864 	 eval_auc: 0.827 	 eval_acc: 0.747 	 test_auc: 0.830 	 test_acc: 0.747 		[0.25, 0.36, 0.54, 0.56, 0.56, 0.63, 0.63, 0.63]
book	train_auc: 0.934 	 train_acc: 0.870 	 eval_auc: 0.744 	 eval_acc: 0.675 	 test_auc: 0.745 	 test_acc: 0.681 		[0.1, 0.19, 0.31, 0.37, 0.37, 0.44, 0.45, 0.45]
ml	train_auc: 0.936 	 train_acc: 0.861 	 eval_auc: 0.892 	 eval_acc: 0.816 	 test_auc: 0.895 	 test_acc: 0.819 		[0.14, 0.25, 0.5, 0.53, 0.53, 0.62, 0.65, 0.72]
yelp	train_auc: 0.892 	 train_acc: 0.798 	 eval_auc: 0.839 	 eval_acc: 0.769 	 test_auc: 0.839 	 test_acc: 0.768 		[0.1, 0.19, 0.34, 0.37, 0.37, 0.42, 0.44, 0.45]

'''