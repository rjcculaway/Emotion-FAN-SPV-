import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import random
from basic_code import load, util, networks
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = util.Logger('./log/','fan_mead')
def main():
    parser = argparse.ArgumentParser(description='PyTorch Frame Attention Network Training')
    parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',
                        help= '0 is self-attention; 1 is self + relation-attention')
    parser.add_argument('--epochs', default=180, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=4e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()
    best_acc = 0
    at_type = ['self-attention', 'self_relation-attention'][args.at_type]
    logger.print('The attention method is {:}, learning rate: {:}'.format(at_type, args.lr))
    
    batchsize_train= 48
    batchsize_eval= 64
    ''' Load model '''
    _structure = networks.resnet18_at(at_type=at_type)
    parameters_directory = './model/'
    parameters = os.listdir(parameters_directory)
    parameters.sort()
    print(os.path.join(parameters_directory, parameters[-1]))
    _parameterDir = os.path.join(parameters_directory, parameters[-1])
    model = load.model_parameters(_structure, _parameterDir)
    ''' Loss & Optimizer '''
    cudnn.benchmark = True
    ''' Train & Eval '''
    if args.evaluate == True:
        logger.print('args.evaluate: {:}', args.evaluate)        
        val(model, at_type, batchsize_eval, args)
        return

def val(model, at_type, batchsize_eval, args):
    topVideo = util.AverageMeter()
    # switch to evaluate mode
    model.eval()
    
    num_of_correct_predictions = 0
    num_of_predictions = 0
    
    video_root = './data/face/eval_mead/'

    identities = [os.path.basename(dir) for dir in os.listdir(video_root)]
    random.shuffle(identities)
    with torch.no_grad():
      for identity in identities:
        videos = os.listdir(os.path.join(video_root, identity))
        for video in videos:
          output_store_fc = []
          output_alpha  = []
          target_store = []
          index_vector = []
          root_single_eval = os.path.join(video_root, identity, video)
          val_loader = load.mead_faces_fan_single(root_single_eval, batchsize_eval)
        # index = np.concatenate(index, axis=0)
          num_of_predictions += 1
          for i, (path, input_var, target, index) in enumerate(val_loader):
              print("Predicting ", path[0].split('/')[5])
              # compute output
              target = target.to(DEVICE)
              input_var = input_var.to(DEVICE)
              ''' model & full_model'''
              f, alphas = model(input_var, phrase = 'eval')
              output_store_fc.append(f)
              output_alpha.append(alphas)
              target_store.append(target)
              index_vector.append(index)

          index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
          index_matrix = []
          for i in range(int(max(index_vector)) + 1):
              index_matrix.append(index_vector == i)

          index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE).float()  # [21570]  --->  [380, 21570]
          output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]
          output_alpha    = torch.cat(output_alpha, dim=0)     # [256,1] ... [256,1]  --->  [21570, 1]
          target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
          ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
          weight_sourcefc = output_store_fc.mul(output_alpha)   #[21570,512] * [21570,1] --->[21570,512]
          sum_alpha = index_matrix.mm(output_alpha) # [380,21570] * [21570,1] -> [380,1]
          weightmean_sourcefc = index_matrix.mm(weight_sourcefc).div(sum_alpha)
          target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
              index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]
          if at_type == 'self-attention':
              pred_score = model(vm=weightmean_sourcefc, phrase='eval', AT_level='pred')
          if at_type == 'self_relation-attention':
              pred_score  = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha, index_matrix=index_matrix, phrase='eval', AT_level='second_level')
          
          print(at_type)
          
          labeller = load.cate2label['MEAD']
          prediction = pred_score.cpu().topk(1)[1].item()
          actual = target_vector.cpu().item()
          print("Prediction: " + labeller[prediction] + "\tActual: " + labeller[actual])
          
          if prediction == actual:
            num_of_correct_predictions += 1
          
          acc_video = util.accuracy(pred_score.cpu(), target_vector.cpu(), topk=(1,))
          topVideo.update(acc_video[0], i + 1)
          logger.print(' *Acc@Video {topVideo.avg:.3f} '.format(topVideo=topVideo), args.evaluate)
          
    print("Overall accuracy:\t" + str(num_of_correct_predictions / num_of_predictions * 100.0))
    return topVideo.avg
if __name__ == '__main__':
    main()
