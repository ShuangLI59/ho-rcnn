import pdb
import numpy as np

def VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt_bbox, min_overlap, cls):
	
	npos = 0
	gt = {}

	for i in range(len(gt_bbox)):
		gt[i] = {}
		if len(gt_bbox[i])>0:
			gt[i]['bb'] = gt_bbox[i]
			gt[i]['det'] = np.zeros([gt_bbox[i].shape[0]])
			npos = npos + gt_bbox[i].shape[0]
		else:
			gt[i]['bb'] = []
			gt[i]['det'] = []

	# % sort detections by decreasing confidence
	si = np.argsort(det_conf)[::-1]
	det_id = det_id[si]
	det_bb = det_bb[si, :]

	# % assign detections to ground truth objects
	nd = len(det_conf)
	tp = np.zeros([nd])
	fp = np.zeros([nd])

	for d in range(nd):
		i = det_id[d]
		bb_1 = det_bb[d, :4]
		bb_2 = det_bb[d, 4:]

		ov_max = -np.inf
		for j in range(len(gt[int(i)]['bb'])):
			# % get gt box
			bbgt_1 = gt[int(i)]['bb'][j, :4]
			bbgt_2 = gt[int(i)]['bb'][j, 4:]

			# % compare box 1
			bi_1 = [np.max([bb_1[0], bbgt_1[0]]), np.max([bb_1[1], bbgt_1[1]]), np.min([bb_1[2], bbgt_1[2]]), np.min([bb_1[3], bbgt_1[3]])]
			iw_1 = bi_1[2]-bi_1[0]+1
			ih_1 = bi_1[3]-bi_1[1]+1

			if iw_1 > 0 and ih_1 > 0:
				# % compute overlap as area of intersection / area of union
				ua_1 = (bb_1[2]-bb_1[0]+1)*(bb_1[3]-bb_1[1]+1) + (bbgt_1[2]-bbgt_1[0]+1)*(bbgt_1[3]-bbgt_1[1]+1) - iw_1*ih_1
				ov_1 = iw_1*ih_1/ua_1
			else:
				ov_1 = 0

			# % compare box 2
			bi_2 = [np.max([bb_2[0], bbgt_2[0]]), np.max([bb_2[1], bbgt_2[1]]), np.min([bb_2[2], bbgt_2[2]]), np.min([bb_2[3], bbgt_2[3]])]
			iw_2 = bi_2[2]-bi_2[0]+1
			ih_2 = bi_2[3]-bi_2[1]+1
			if iw_2 > 0 and ih_2 > 0:
				# % compute overlap as area of intersection / area of union
				ua_2 = (bb_2[2]-bb_2[0]+1)*(bb_2[3]-bb_2[1]+1) + (bbgt_2[2]-bbgt_2[0]+1)*(bbgt_2[3]-bbgt_2[1]+1) - iw_2*ih_2
				ov_2 = iw_2*ih_2/ua_2
			else:
				ov_2 = 0

			# % get minimum
			min_ov = np.min([ov_1, ov_2])
			# % update ov_max & j_max
			if min_ov > ov_max:
				ov_max = min_ov
				j_max = j

		# % assign detection as true positive/don't care/false positive
		if ov_max >= min_overlap:
			if not gt[int(i)]['det'][j_max]:
				tp[d] = 1
				gt[int(i)]['det'][j_max] = True
			else:
				fp[d] = 1
		else:
			fp[d] = 1

	# % compute precision/recall
	fp = np.cumsum(fp)
	tp = np.cumsum(tp)
	rec = tp/npos
	prec = tp/(fp+tp)

	# % compute average precision
	ap = 0
	for t in np.arange(0,1,0.1):
		if len(prec[rec >= t])==0:
			p = 0
		else:
			p = np.max(prec[rec >= t])
		ap = ap + p/11
	    
	return rec, prec, ap

