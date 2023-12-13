import os
import time
import cv2
import numpy as np
import metrics as M
from multiprocessing import Pool
def cal_metrics(pred_root,gt_root,dataset,log="log.txt"):
    gt_dataset = os.path.join(gt_root,dataset)
    pred_dataset = os.path.join(pred_root,dataset)
    eval_folder = pred_root.split(os.path.sep)[-1]

    sm=0.
    wfm=0.
    mae=0.
    adpEm=0.
    meanEm=0.
    adpFm=0.
    meanFm=0.
    maxFm=[]
    maxEm=[]

    f = open(log,"a")
    for each_video_gt in sorted(os.listdir(gt_dataset)):

        FM1 = M.Fmeasure()
        WFM1 = M.WeightedFmeasure()
        SM1= M.Smeasure()
        EM1 = M.Emeasure()
        MAE1 = M.MAE()

        gt_name_list = sorted(os.listdir(os.path.join(gt_dataset,each_video_gt,"GT_object_level")))
        gt_name_list = gt_name_list[1:-1]

        for gt_name_now in gt_name_list:
            gt_path_now = os.path.join(gt_dataset,each_video_gt, "GT_object_level",gt_name_now)
            pred_path_now = os.path.join(pred_dataset,each_video_gt,gt_name_now)

            gt = cv2.imread(gt_path_now, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path_now, cv2.IMREAD_GRAYSCALE)

            FM1.step(pred=pred, gt=gt)
            WFM1.step(pred=pred, gt=gt)
            SM1.step(pred=pred, gt=gt)
            EM1.step(pred=pred, gt=gt)
            MAE1.step(pred=pred, gt=gt)

        fm1 = FM1.get_results()['fm']
        wfm1 = WFM1.get_results()['wfm']
        sm1 = SM1.get_results()['sm']
        em1 = EM1.get_results()['em']
        mae1 = MAE1.get_results()['mae']
        sm += sm1
        maxFm.append(fm1['curve'])
        mae += mae1
        wfm += wfm1
        meanEm += 0 if em1['curve'] is None else em1['curve'].mean()
        maxEm.append(em1['curve'])
        adpFm += fm1['adp']
        meanFm += fm1['curve'].mean()

    length = len(os.listdir(gt_dataset))
    sm/=length
    wfm/=length
    mae/=length
    meanEm/=length

    adpFm/=length
    meanFm/=length

    maxFm = np.array(maxFm)
    maxF = max(maxFm.mean(0))

    maxEm = np.array(maxEm)
    maxE = max(maxEm.mean(0))

    print(
        'eval:' + eval_folder + ',' +
        'Dataset:', dataset, ',',
        'Smeasure:', round(sm,4), '; ',
        'MAE:', round(mae,4), '; ',
        'maxFm:', round(maxF,4),
        'maxEm:', round(maxE,4), '; ',
        'meanFm:', round(meanFm,4), '; ',
        'meanEm:', round(meanEm,4), '; ',
        'wFmeasure:', round(wfm,4), '; ',
       # 'adpEm:', adpEm, '; ',
        'adpFm:', round(adpFm,4), '; \n',

        sep=''
    )
    f.write(
        'eval:' + eval_folder + ',' +
        'Dataset:' + dataset + ',' +
        'Smeasure:' + str(round(sm,4)) + '; ' +
        'MAE:' + str(round(mae,4)) + '; ' +
        'maxFm:' + str(round(maxF,4)) + '; '+
        'maxEm:' + str(round(maxE,4)) + '; ' +
        'meanFm:' + str(round(meanFm,4)) + '; ' +
        'meanEm:' + str(round(meanEm,4)) + '; ' +
        'wFmeasure:' + str(round(wfm,4)) + '; ' +
        'adpFm:' + str(round(adpFm,4)) + '; ' +
        "\n")
    f.close()
    return {dataset:mae}

def eval_all_datasets(pred_root, gt_root, datasets,log="log.txt"):
    a = time.perf_counter()
    pool = Pool(processes=9)
    metrics = pool.starmap(cal_metrics, [[pred_root,gt_root,dataset,log] for dataset in datasets])
    b = time.perf_counter()
    print("eval time:",b - a)
    return metrics

if __name__ == '__main__':
    eval_all_datasets(pred_root="/path/to/saliency/maps/dir",gt_root="/path/to/dataset/test",datasets=['VISAL', 'DAVSOD', 'DAVIS', 'FBMS'],log="metrics_log.txt")

