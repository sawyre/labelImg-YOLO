#!/usr/bin/env python
# -*- coding: utf-8 -*-

from operator import itemgetter

def calc_mean_ap(input_zip):
    pos_list = [0.1*x for x in range(0, 10)]
    #print pos_list
    return_zip = []
    mAP = 0
    for it in pos_list:
        precisions = [x[1] for x in input_zip if x[0]>it]
        if len(precisions)<=0:
            pr=0
        else:
            pr = max(precisions)
        mAP+=pr*0.1
    return mAP
#return return_zip

def calc_recall_precision(sc, y):
	sc_list = zip(sc, y)
	sc_list_s = sorted(sc_list, key=itemgetter(0), reverse=True)
	
	#print sc_list_s
	
	precision = []
	recall = []

	s = []
	gt = []
	precision.append(0.0)
	recall.append(0.0)
	arr_s = len(sc_list_s)

	false_pos = 0
	true_pos = 0
	for pair in sc_list_s:
		curr_score = pair[0]
		curr_y = pair[1]
		if curr_y==1:
			true_pos+=1
		else:
			false_pos+=1
	
		false_neg = 0
		for p in sc_list_s:
			sc = p[0]
			py = p[1]
			if sc<curr_score and py==1:
				false_neg+=1
		curr_pr = float(true_pos)/(true_pos+false_pos)
		
		if (true_pos+false_neg)==0:
		    curr_rec=0.0
		else:
		    curr_rec = float(true_pos)/(true_pos+false_neg)
		precision.append(curr_pr)
		recall.append(curr_rec)

	prc = zip(recall, precision)
	return prc

def calc_auc(input_zip):
	indices = [x for x in range(0, len(input_zip)-1)]
	auc = 0
	for ind in indices:
		w = - input_zip[ind][0] + input_zip[ind+1][0]
		h1 = input_zip[ind][1]
		h2 = input_zip[ind+1][1]
		#print w, h1, h2
		minh = min(h1, h2)
		maxh = max(h2, h2)
		rectangleS = w*minh
		triangleS = (w*(maxh-minh))/2
		currS = rectangleS+triangleS
		auc+=currS
	return auc
	
'''
scores = [0.8, 0.5, 0.39, 0.7, 0, 0]
y = [0,1,0,0,1,1]

prc = calc_recall_precision(scores, y)
print prc

mAP = calc_mean_ap(prc)
	
print mAP
#print result_zip
'''
