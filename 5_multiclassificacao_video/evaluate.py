import numpy as np
import tensorflow as tf



def calc_metrics(Y, num_classes, output):

    sAP_accum = 0
    hit1_accum = 0
    hit5_accum = 0
    hit20_accum = 0  

    for index1,sample in enumerate(output):
        aux = np.arange(num_classes)

        ground_truth = []

        for index2,value in enumerate(Y[index1]):
            if value == 1:
                ground_truth.append(index2)

        top20_pos = aux[np.argsort(-sample)]
        top20_sample = sample[np.argsort(-sample)]

        is_top1 = False
        is_top5 = False
        is_top20 = False

        calc =  np.ndarray(shape=(num_classes), dtype=float)
        
        correct_counter = 0
        for index, prediction in enumerate(top20_pos):
            calc[index] = 0
            for label in ground_truth:
                if prediction == label:
                    calc[index] = 1.0 / (index+1)
                    correct_counter += 1

                    if index == 0:
                        is_top1 = True
                        is_top5 = True
                        is_top20 = True
                    elif index < 5:
                        is_top5 = True
                        is_top20 = True
                    elif index < 20:
                        is_top20 = True     

                    break

        if correct_counter > 0:
            sAP = np.sum(calc)/correct_counter
            sAP_accum += sAP
            #print(calc, correct_counter, sAP)
        
        #print(calc)

        if is_top1 == True:
            hit1_accum += 1
        if is_top5 == True:
            hit5_accum += 1  
        if is_top20 == True:
            hit20_accum += 1  

              

    mAP = sAP_accum / Y.shape[0]

    top1 = hit1_accum / Y.shape[0]
    top5 = hit5_accum / Y.shape[0]
    top20 = hit20_accum / Y.shape[0]
  
    return mAP, top1, top5, top20


mAP = 0
top1 = 0
top5 = 0
top20 = 0
val_batch_counter = 0

def register_batch_evaluation(labels, num_classes, result):
    global mAP, top1, top5, top20, val_batch_counter 

    batch_mAP, batch_top1, batch_top5, batch_top20 = calc_metrics(labels, num_classes, result)
    mAP += batch_mAP
    top1 += batch_top1
    top5 += batch_top5
    top20 += batch_top20

    val_batch_counter += 1


def get_global_evaluation_result():
    global mAP, top1, top5, top20, val_batch_counter 

    global_acc = mAP/val_batch_counter
    global_top1 = top1/val_batch_counter
    global_top5 = top5/val_batch_counter
    global_top20 = top20/val_batch_counter

    return global_acc, global_top1, global_top5, global_top20

def clear_registered_evaluations():
    global mAP, top1, top5, top20, val_batch_counter 
    mAP = 0
    top1 = 0
    top5 = 0
    top20 = 0
    val_batch_counter = 0