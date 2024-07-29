from csv import DictWriter
from os.path import join, isfile
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
import torch
from .evaluation import specificity

def make_inferences(model, data_loader, fp16_scaler, instances):

    model.eval()
    y_pred, y_true, y_score = [], [], []

    with torch.no_grad():
        for _, data in enumerate(tqdm(data_loader, desc="inferences on test set")):
            bscan_ti, bscan_tj = map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[:2])
            side_eye, bscan_num, sex, age, delta_h = map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[2:7])
            labels = data[7].cuda(non_blocking=True).type(torch.int64)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                logits = model(bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h)

            torch.cuda.synchronize()

            x = torch.softmax(logits, dim=1)
            y_score.extend(x.squeeze().tolist())
            y_pred.extend(torch.argmax(x, dim=1).tolist())
            y_true.extend(labels.squeeze().tolist())


    raw_output = {  'case_id': list(instances),
                    'y_pred': y_pred,
                    'y_true': y_true,
                    'y_score': y_score
                    }
    
    return raw_output

def export_results(raw_output, classes, date, output_dir):

    try:
        os.mkdir(output_dir)
    except:
        print('output dir already exist')

    df_global = pd.DataFrame.from_dict(raw_output)
    df_global.to_csv(join(output_dir, f'dataframe_output_{date}.csv'))

    y_pred, y_true, y_score = raw_output['y_pred'], raw_output['y_true'], raw_output['y_score']
    save_csv_metrics(get_metrics_skm(y_pred, y_true, y_score), output_dir, date)
    save_conf_matrix(y_pred, y_true, classes, output_dir, date)

def get_metrics_skm(y_pred, y_true, y_score):
    return  {
            "Rk-correlation": skm.matthews_corrcoef(y_true, y_pred),
            "Specificity": specificity(y_true, y_pred),
            'f1score': skm.f1_score(y_true, y_pred, average="micro"),
            'accuracy': skm.accuracy_score(y_true, y_pred),
            'precision': skm.precision_score(y_true, y_pred, average="micro"),
            'recall': skm.recall_score(y_true, y_pred, average="micro"),
            'kappa': skm.cohen_kappa_score(y_true, y_pred)
            }

def save_conf_matrix(y_pred, y_true, classes, output_dir, date):
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * len(classes), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.rcParams['figure.figsize'] = [15, 11]
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True)
    plt.savefig(join(output_dir, f'confusion_testset_unimodal_{date}.png'), dpi=300)
    plt.clf()

def save_csv_metrics(metrics, output_dir, date):
    csv_name = join(output_dir, f'metricsCSV_unimodal_{date}.csv')
    exists = isfile(csv_name)
    with open(csv_name, 'a+') as f:
        header = list(metrics.keys())
        writer = DictWriter(f, delimiter=',', lineterminator='\n',fieldnames=header)

        if not exists:
            writer.writeheader()

        writer.writerow(metrics)
        f.close()