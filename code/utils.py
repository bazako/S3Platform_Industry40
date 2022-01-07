import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def calculate_eer(labels, scores):
	norm='MinMax'
	pos_label=0
	
	if norm == 'MixMax':
		scores=(scores-min(scores))/(max(scores)-min(scores))
	scores = 1 - scores

	fpr_global, tpr_global, threshold_global = roc_curve(labels,scores, pos_label=pos_label)
	fnr_global = 1 - tpr_global
	# eer_threshold = threshold_global[np.nanargmin(np.absolute((fnr_global - fpr_global)))]
	idx_eer=np.nanargmin(np.absolute((fnr_global - fpr_global)))
	fpr_eer=fpr_global[idx_eer]
	fnr_eer=fnr_global[idx_eer]
	
	
	eer_score = 0.5*(fpr_eer+fnr_eer)

	return eer_score
	
def calculate_metrics(scores, labels, norm='MinMax',pos_label=1):

	if scores.shape[1] > 1:
		scores=scores[:,0]
	
	if norm == 'MixMax':
		scores=(scores-min(scores))/(max(scores)-min(scores))
	
	fpr_global, tpr_global, threshold_global = roc_curve(labels,scores, pos_label=pos_label)
	fnr_global = 1 - tpr_global
	eer_threshold = threshold_global[np.nanargmin(np.absolute((fnr_global - fpr_global)))]
	eer_score = calculate_eer(labels,scores)
	auc_score = auc(fpr_global, tpr_global)
	
	precision,recall,threshol=precision_recall_curve(labels,scores, pos_label=pos_label)
	F1=2*precision*recall/(precision+recall)
	f1max=max(F1)
	return eer_score, auc_score, f1max, eer_threshold

