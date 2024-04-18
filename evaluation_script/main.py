import numpy as np
import json
import pycocotools.mask as mask_utils

def ann_to_rle(ann, h, w):
    """Convert annotation which can be polygons, uncompressed RLE to RLE.
    Args:
        ann (dict) : annotation object
    Returns:
        ann (rle)
    """

    segm = ann
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann
    return rle

def ann_to_mask(seg):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object
        Returns:
            binary mask (numpy 2D array)
        """
        ann = seg
        height, width = seg['size'][0], seg['size'][1]
        rle = ann_to_rle(ann, height, width)
        return mask_utils.decode(rle)

def masks_to_res_dict(masks, num_classes, obj=False):
    res_dict = {}
    for mask in masks:
        file_name = mask['file_name']
        category_id = mask['category_id']
        img_size = mask['segmentation']['size']
        if obj:
            file_name = f'{file_name}__{category_id}'
        if file_name not in res_dict:
            res_dict[file_name] = np.zeros(img_size, dtype=np.float) + num_classes
        
        if category_id >= num_classes:
            continue
        bmask = ann_to_mask(mask['segmentation'])
        # print(file_name, category_id)
        # print(res_dict[file_name].shape, bmask.shape, category_id)
        res_dict[file_name][bmask==1] = category_id
    return res_dict


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    num_classes = 116
    class_names = ["aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", "bicycle's wheel", \
                    "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", "bird's wing", "bird's tail", "bird's head", "bird's eye",\
                    "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", "bottle's body", "bottle's cap", "bus's wheel", "bus's headlight", "bus's front", \
                    "bus's side", "bus's back", "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", "car's wheel", "car's headlight", "car's front", \
                    "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", "cat's tail", "cat's head", "cat's eye", "cat's torso",\
                    "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", \
                    "cow's horn", "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", "horse's tail", "horse's head", \
                    "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", "horse's muzzle", "horse's hoof", "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", \
                    "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", \
                    "person's upper arm", "person's hand", "pottedplant's pot", "pottedplant's plant", "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", \
                    "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", "train's coach", "tvmonitor's screen"]
    evaluation_set = {'base': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 42, \
                               43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 70, 71, 72, 73, 74, \
                                75, 76, 77, 78, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 108, 109, 110, 111, 112, 113, 114, 115]}
    gt_masks = json.load(open(test_annotation_file, 'r'))
    gt_part_masks = gt_masks['part_sem_seg_gt']
    gt_part_mask_dict = masks_to_res_dict(gt_part_masks, num_classes)     
    
    pred_part_masks = json.load(open(user_submission_file, 'r'))
    pred_part_mask_dict = masks_to_res_dict(pred_part_masks, num_classes)  

    conf_matrix = np.zeros(((num_classes + 1), (num_classes + 1))).astype(int)

    gt_obj_masks = gt_masks['object_sem_seg_gt']#json.load(open(test_annotation_file.replace('part', 'obj'), 'r'))
    
    for obj_mask in gt_obj_masks:
        file_name = obj_mask['file_name']
        file_name = file_name.split('__')[0]
        obj_binary_mask = ann_to_mask(obj_mask['segmentation'])
        gt = gt_part_mask_dict[file_name].astype(int)
        gt[obj_binary_mask==0] = num_classes
        pred = pred_part_mask_dict[file_name].astype(int)
        pred[obj_binary_mask==0] = num_classes

        conf_matrix += np.bincount(
                (num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=conf_matrix.size,
            ).reshape(conf_matrix.shape)
    
    acc = np.full(num_classes, np.nan, dtype=np.float)
    iou = np.full(num_classes, np.nan, dtype=np.float)
    tp = conf_matrix.diagonal()[:-1].astype(np.float)
    
    pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float)
    class_weights = pos_gt / np.sum(pos_gt)
    pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)

    res = {}
    res["mIoU"] = 100 * miou
    for i, name in enumerate(class_names):
        res["IoU-{}".format(name)] = 100 * iou[i]
    for set_name, set_inds in evaluation_set.items():
        iou_list = []
        set_inds = np.array(set_inds, int)
        mask = np.zeros((len(iou),)).astype(np.bool)
        mask[set_inds] = 1
        miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
        pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
        res["mIoU-{}".format(set_name)] = 100 * miou
        # output["pAcc-{}".format(set_name)] = 100 * pacc
        iou_list.append(miou)
        miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
        pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
        res["mIoU-un{}".format(set_name)] = 100 * miou
        # output["pAcc-un{}".format(set_name)] = 100 * pacc
        iou_list.append(miou)
    res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase'])
    
    output = {}
    if phase_codename == "dev1":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "test_split": {
                    "h-IoU": res['h-IoU'],
                    "mIoU": res["mIoU"],
                    "mIoU-base": res["mIoU-base"],
                    "mIoU-unbase": res["mIoU-unbase"],
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "dev2":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "mIoU": res["mIoU"],
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")
    return output