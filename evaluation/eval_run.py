import pdb
import os
import numpy as np
import scipy.io as sio
from get_list_coco_obj import list_coco_obj
from VOCevaldet_bboxpair import VOCevaldet_bboxpair

min_overlap = 0.5;
base_dir =  '/data/vision/torralba/ls-objectvideo/1adobe/1codes/ho-rcnn';
caffe_root = os.path.join(base_dir, 'caffe/');
frcnn_root = os.path.join(base_dir, 'fast-rcnn/');

im_root   = os.path.join(base_dir, 'data/hico_20160224_det/images/');
anno_file = os.path.join(base_dir, 'data/hico_20160224_det/anno.mat');
bbox_file = os.path.join(base_dir, 'data/hico_20160224_det/anno_bbox.mat');


def eval_one(image_set, iter, eval_mode, anno, bbox):
	
	exp_name = 'rcnn_caffenet_ho_pconv_ip1_s'  
	exp_dir = 'ho_1_s'  
	prefix = 'rcnn_caffenet_pconv_ip'  
	format = 'obj'
	score_blob = 'n/a'

	# set detection root
	det_root = '%s/output/%s/hico_det_%s/%s_iter_%d/' % (base_dir, exp_dir, image_set, prefix, iter)

	# set res file
	res_root = '%s/evaluation/result/%s/' % (base_dir, exp_name)
	res_file = '%s%s_%s_%06d.mat' % (res_root, eval_mode, image_set, iter)
	os.makedirs(res_root, exist_ok=True)

	# % get gt bbox
	if image_set=='train2015':
	    gt_bbox = bbox['bbox_train']
	    list_im = anno['list_train']
	    anno_im = anno['anno_train']
	elif image_set=='test2015':
	    gt_bbox = bbox['bbox_test'][0]
	    list_im = anno['list_test']
	    anno_im = anno['anno_test']
	else:
	    error('image_set error\n')
	assert(len(gt_bbox) == len(list_im))

	# % copy variables
	list_action = anno['list_action']
	num_action = len(list_action)
	num_image = len(gt_bbox)


	# % get HOI index intervals for object classes
	obj_hoi_int = np.zeros([len(list_coco_obj), 2])
	for i in range(len(list_coco_obj)):
		hoi_int = [j for j, tem in enumerate(list_action) if list_coco_obj[i] in str(tem[0][0][0])]
		assert len(hoi_int)>0
		obj_hoi_int[i, 0] = hoi_int[0]
		obj_hoi_int[i, 1] = hoi_int[-1]
	obj_hoi_int = obj_hoi_int.astype(int)
	

	print('start evaluation')
	print('setting:     %s' % eval_mode)
	print('exp_name:    %s' % exp_name)
	print('score_blob:  %s' % score_blob)
	print('\n')

	if os.path.exists(res_file):
		# % load result file
	    print('results loaded from %s\n', res_file);
	    ld = sio.loadmat(res_file)
	    AP = ld['AP']
	    REC = ld['REC']
	    # % print ap for each class
	    for i in range(num_action):
	        nname = list_action[i][0][0][0]
	        aname = '%s_%s' % (list_action[i][0][2][0], list_action[i][0][0][0])
	        print('  %03d/%03d %-30s ap: %.4f  rec: %.4f\n' % (i, num_action, aname, AP[i], REC[i]))
	else:
		gt_all = {}
		print('converting gt bbox format ...')
		for i in range(num_image):
			assert gt_bbox[i][0][0]==list_im[i][0][0]
			for j in range(len(gt_bbox[i][2][0])):
				if not gt_bbox[i][2][0][j][4][0][0]:
					hoi_id = gt_bbox[i][2][0][j][0][0][0]-1
					bbox_h = gt_bbox[i][2][0][j][1][0]
					bbox_o = gt_bbox[i][2][0][j][2][0]
					conn = gt_bbox[i][2][0][j][3]-1
					boxes = np.zeros([conn.shape[0], 8])

					for k in range(conn.shape[0]):
						boxes[k, 0] = bbox_h[conn[k][0]][0][0][0]
						boxes[k, 1] = bbox_h[conn[k][0]][2][0][0]
						boxes[k, 2] = bbox_h[conn[k][0]][1][0][0]
						boxes[k, 3] = bbox_h[conn[k][0]][3][0][0]

						boxes[k, 4] = bbox_o[conn[k][1]][0][0][0]
						boxes[k, 5] = bbox_o[conn[k][1]][2][0][0]
						boxes[k, 6] = bbox_o[conn[k][1]][1][0][0]
						boxes[k, 7] = bbox_o[conn[k][1]][3][0][0]
				
	
					# assert (hoi_id, i) not in gt_all
					# gt_all[(hoi_id, i)] = boxes

					if hoi_id not in gt_all:
						gt_all[hoi_id] = {}
					# if i not in gt_all[hoi_id]:
					assert i not in gt_all[hoi_id]
					gt_all[hoi_id][i] = {}
					gt_all[hoi_id][i] = boxes

		gt_all_new = {}
		for ii in range(num_action):
			gt_all_new[ii] = [[] for _ in range(num_image)]
			if ii in gt_all:
				for jj in range(num_image):
					if jj in gt_all[ii]:
						gt_all_new[ii][jj] = gt_all[ii][jj]
		gt_all = gt_all_new
		print('done.')


	
		if format=='obj':
			all_boxes = np.zeros([num_action, 1])
		elif format=='all':
			det_file = '%s/detections.mat' % det_root
			ld = sio.loadmat(det_file)
			all_boxes = ld['all_boxes']
			pdb.set_trace()

		# % compute ap for each class
		AP = np.zeros([num_action, 1])
		REC = np.zeros([num_action, 1])
		print('start computing ap ...')

		for i in range(num_action):
			nname = list_action[i][0][0][0]
			aname = '%s_%s' % (list_action[i][0][2][0], list_action[i][0][0][0])
			print('  %03d/%03d %-30s' % (i, num_action, aname))

			if format=='obj':
				# % get object id and action id within the object category
				obj_id = [index for index, tem in enumerate(list_coco_obj) if nname==tem]
				assert len(obj_id)==1
				
				obj_id = obj_id[0]
				# act_id = i - obj_hoi_int[obj_id, 0] + 1
				act_id = i - obj_hoi_int[obj_id, 0]
				det_file = '%s/detections_' % det_root +'%02d'%(obj_id+1)+'.mat'

				ld = sio.loadmat(det_file)
				det = ld['all_boxes'][act_id]

			elif format=='all':
				det = all_boxes[i, :]

			# % convert detection results
			det_id = []
			det_bb = []
			det_conf = []

			for j in range(len(det)):
				if det[j].shape[0]!=0:
					num_det = det[j].shape[0]

					if len(det_id)==0:
						det_id = j * np.ones([num_det, 1])
					else:
						det_id = np.concatenate((det_id, j * np.ones([num_det, 1])), axis=0)

					if len(det_bb)==0:
						det_bb = det[j][:, :8]
					else:
						det_bb = np.concatenate((det_bb, det[j][:, :8]), axis=0)

					if len(det_conf)==0:
						det_conf = det[j][:, 8]
					else:
						det_conf = np.concatenate((det_conf, det[j][:, 8]), axis=0)
			
			# % convert zero-based to one-based indices
			det_bb = det_bb + 1
	        # % get gt bbox
			assert(len(det) == len(gt_bbox))
			gt = gt_all[i]

			if eval_mode=='def':
				pass
			elif eval_mode=='ko':
				nid = [index for index, tem in enumerate(list_action) if tem[0][0][0] == nname]

				iid = np.where(np.any(anno_im[nid, :] == 1, axis=0)==1)[0]
				
				keep = []
				for tem in det_id:
					if int(tem) in iid:
						keep.append(True)
					else:
						keep.append(False)

				det_id = det_id[keep]
				det_bb = det_bb[keep, :]
				det_conf = det_conf[keep]

			rec, prec, ap = VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt, min_overlap, aname)
			AP[i] = ap
			try:
				REC[i] = rec[-1]
			except:
				pdb.set_trace()
			# print('  ap: %.4f  rec: %.4f' % (ap, REC[i]))

			# break

		print('done.')
		# save(res_file, 'AP', 'REC');


	# % get number of instances for each class
	num_inst = np.zeros([num_action, 1])
	
	for i in range(len(bbox['bbox_train'][0])):
		for j in range(len(bbox['bbox_train'][0][i][2][0])):
			if not bbox['bbox_train'][0][i][2][0][j][4][0][0]:
				hoi_id = bbox['bbox_train'][0][i][2][0][j][0][0][0]-1
				num_inst[hoi_id] = num_inst[hoi_id] + bbox['bbox_train'][0][i][2][0][j][3].shape[0]

	s_ind = num_inst < 10
	p_ind = num_inst >= 10


	print('setting:     %s' % eval_mode)
	print('exp_name:    %s' % exp_name)
	print('score_blob:  %s' % score_blob)
	print('  mAP / mRec (full):      %.4f / %.4f' % (np.mean(AP), np.mean(REC)))
	print('  mAP / mRec (rare):      %.4f / %.4f' % (np.mean(AP[s_ind]), np.mean(REC[s_ind])))
	print('  mAP / mRec (non-rare):  %.4f / %.4f' % (np.mean(AP[p_ind]), np.mean(REC[p_ind])))



def eval_run():
	image_set = 'test2015';
	iter = 150000;
	
	# load annotations
	anno = sio.loadmat(anno_file)
	bbox = sio.loadmat(bbox_file)

	eval_mode = 'def'  
	eval_one(image_set, iter, eval_mode, anno, bbox)
	
	eval_mode = 'ko'   
	eval_one(image_set, iter, eval_mode, anno, bbox)


if __name__ == '__main__':
    eval_run()
