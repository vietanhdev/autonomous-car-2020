import numpy as np
import sys
import cv2
sys.path.append('.')
try:
    from lib.core.model.facebox.utils.box_utils import encode, iou, decode
except:
    from utils.box_utils import encode, iou

from lib.lib_config import config as cfg
# from lib.helper.logger import logger

def get_training_targets(groundtruth_boxes, threshold=0.5,anchors=cfg.MODEL.anchors):


    """
    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 5].
        threshold: a float number.
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
        cls_targets: an int tensor with shape [num_anchors, 1], possible values
            that it can contain class value of each anchor.
    """
    reg_targets=np.zeros(shape=[cfg.MODEL.num_anchors,4])
    cls_targets = np.zeros(shape=[cfg.MODEL.num_anchors]).astype(np.int32)
    if len(groundtruth_boxes) > 0:
        boxes_ = groundtruth_boxes[:, 0:4]
        N = np.shape(groundtruth_boxes)[0]
        num_anchors = np.shape(anchors)[0]
        no_match_tensor = np.ones(shape=[num_anchors])*-1

        if N>0:
            matches=_match(anchors, boxes_, threshold)
        else:
            matches=no_match_tensor


        matches = np.array(matches,dtype=np.int)


        reg_targets, cls_targets = _create_targets(
        anchors, groundtruth_boxes, matches
        )

    return reg_targets, cls_targets


def _match(anchors, groundtruth_boxes, threshold=0.5):
    """Matching algorithm:
    1) for each groundtruth box choose the anchor with largest iou,
    2) remove this set of anchors from the set of all anchors,
    3) for each remaining anchor choose the groundtruth box with largest iou,
       but only if this iou is larger than `threshold`.

    Note: after step 1, it could happen that for some two groundtruth boxes
    chosen anchors are the same. Let's hope this never happens.
    Also see the comments below.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        an int tensor with shape [num_anchors].
    """
    num_anchors = np.shape(anchors)[0]

    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou(groundtruth_boxes, anchors)  # shape [N, num_anchors]
    matches = np.argmax(similarity_matrix, axis=0).astype(np.int32)  # shape [num_anchors]

    matched_vals = np.max(similarity_matrix, axis=0)  # shape [num_anchors]

    below_threshold = np.greater(threshold, matched_vals).astype(np.int32)


    matches = np.add(np.multiply(matches, 1 - below_threshold), -1 * below_threshold)

    # after this, it could happen that some groundtruth
    # boxes are not matched with any anchor box

    # now we must ensure that each row (groundtruth box) is matched to
    # at least one column (which is not guaranteed
    # otherwise if `threshold` is high)

    # for each groundtruth box choose the anchor box with largest iou
    # (force match for each groundtruth box)
    forced_matches_ids = np.argmax(similarity_matrix, axis=1)  # shape [N]

    # if all indices in forced_matches_ids are different then all rows will be matched
    #forced_matches_indicators = tf.one_hot(forced_matches_ids, depth=num_anchors, dtype=tf.int32)  # shape [N, num_anchors]
    forced_matches_indicators = np_one_hot(forced_matches_ids, depth=num_anchors)  # shape [N, num_anchors]
    forced_match_row_ids = np.argmax(forced_matches_indicators, axis=0).astype(np.int)  # shape [num_anchors]

    forced_match_mask = np.greater(np.max(forced_matches_indicators, axis=0), 0)  # shape [num_anchors]

    matches = np.where(forced_match_mask, forced_match_row_ids, matches)
    # even after this it could happen that some rows aren't matched,
    # but i believe that this event has low probability

    #print(np.sum(matches[matches>=0]))

    return matches
def np_one_hot(data,depth):
    return (np.arange(depth) == data[:, None]).astype(np.int)


def _create_targets(anchors, groundtruth_boxes, matches):
    """Returns regression targets for each anchor.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 5].
        matches: a int tensor with shape [num_anchors].
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
        cls_targets: an int tensor with shape [num_anchors, 1], possible values
            that it can contain class value of each anchor.
    """
    boxes_ = groundtruth_boxes[:, 0:4]
    klass_ = groundtruth_boxes[:, 4]

    reg_targets=np.zeros(shape=[cfg.MODEL.num_anchors,4])
    cls_targets = np.zeros(shape=[cfg.MODEL.num_anchors]).astype(np.int32)

    matched_anchor_indices = np.array(np.where(np.greater_equal(matches, 0)))  # shape [num_matches, 1]
    matched_anchor_indices = np.squeeze(matched_anchor_indices, axis=0)


    if len(matched_anchor_indices)==0:
        return reg_targets, cls_targets


    matched_gt_indices =matches[matched_anchor_indices] # shape [num_matches]

    matched_anchors = anchors[matched_anchor_indices]  # shape [num_matches, 4]
    matched_gt_boxes = boxes_[matched_gt_indices]  # shape [num_matches, 4]
    matched_reg_targets = encode(matched_gt_boxes, matched_anchors)  # shape [num_matches, 4]

    matched_cls_targets = klass_[matched_gt_indices]  # shape [num_matches, 1]

    for i,index in enumerate(matched_anchor_indices):
        reg_targets[index,:] = matched_reg_targets[i,:]
        cls_targets[index] = int(matched_cls_targets[i])

    return reg_targets, cls_targets


if __name__=='__main__':





    from lib.lib_config import config as cfg

    default_anchors = cfg.MODEL.anchors
    win = cfg.MODEL.win
    hin = cfg.MODEL.hin
    groundtruth_boxes = np.array([[0.32759732, 0.30244141, 0.3535777,  0.32003697, 1],
                                     [0.32459959, 0.32942127 ,0.34058751, 0.34818987, 2],
                                     [0.1 ,0.1   ,0.2, 0.3, 3]]
                                            , dtype=np.float32)

    reg_targets, cls_targets = get_training_targets(
        groundtruth_boxes=groundtruth_boxes,
        threshold=0.35,
        anchors=default_anchors)

    print("default anchor ", default_anchors.shape, default_anchors)
    print(np.where(cls_targets >= 0)[0].shape)
    print("regression targets ", reg_targets.shape, reg_targets)

    img = np.ones((win, hin, 3)).astype(np.uint8) * 255

    # Draw matched anchor boxes
    for i in range(0, default_anchors.shape[0]):
        anchor = default_anchors[i]
        cls = cls_targets[i]
        if cls > 0:
            cv2.rectangle(img,(int(anchor[1]*win), int(anchor[0]*hin)), (int(anchor[3]*win), int(anchor[2]*hin)), (0, 255, 255), 1)
            cv2.putText(img, str(cls), (int(anchor[1]*win), int(anchor[0]*hin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)                        
        
    # Draw groundtruth boxes
    for box in groundtruth_boxes:
        cv2.rectangle(img, (int(box[1]*win), int(box[0]*hin)),
                                            (int(box[3]*win), int(box[2]*hin)), (255, 0, 255), 1)
    

    cv2.imshow('TEST', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

