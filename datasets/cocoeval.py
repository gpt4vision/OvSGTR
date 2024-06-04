import numpy as np

from pycocotools.cocoeval import COCOeval
from .bbox_overlaps import bbox_overlaps

from util.misc import get_rank


class COCOEval(COCOeval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_dt_valid = {}

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]

        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        has_group_of = False
        if len(gt) > 0 and len(dt) > 0 and 'is_group_of' in gt[0]:
            has_group_of = True

        if has_group_of:
            dt_boxes = np.array([d['bbox'] for d in dt]).reshape(-1, 4)
            gt_boxes = np.array([g['bbox'] for g in gt]).reshape(-1, 4)
            dt_boxes[:, 2:] = dt_boxes[:, 2:] + dt_boxes[:, :2]
            gt_boxes[:, 2:] = gt_boxes[:, 2:] + gt_boxes[:, :2]


            is_group_of = np.array([g['is_group_of'] for g in gt], dtype=bool)
            is_group_idx = np.where(is_group_of)[0]
            non_group_idx = np.where(~is_group_of)[0]

            iofs = bbox_overlaps(dt_boxes, gt_boxes, mode='iof')

            non_group_gt = [gt[e] for e in non_group_idx]
            group_gt = [gt[e] for e in is_group_idx]
            # step 1: for non-group-of gts.
            if len(ious) > 0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t,1-1e-10])
                        m   = -1
                        for gind, g in zip(non_group_idx, non_group_gt):
                            # if this gt already matched, and not a crowd, continue
                            if gtm[tind,gind]>0 and not iscrowd[gind]:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                                break

                            # continue to next gt unless better match made
                            if ious[dind,gind] < iou:
                                continue
                            # if match successful and best so far, store appropriately
                            iou=ious[dind,gind]
                            m=gind

                        if m ==-1:
                            continue
                        # if match made store id of match for both dt and gt
                        dtIg[tind,dind] = gtIg[m] 
                        dtm[tind,dind]  = gt[m]['id']
                        gtm[tind,m]     = d['id']       

            # step 2: for group-of gts
            if len(is_group_idx) > 0 and len(iofs) > 0:
                for tind, t in enumerate(p.iouThrs):
                    iof_thresh = min([t,1-1e-10])
                    maxIoF = [-1 for _ in range(len(gt))]  # store maximum IoF for each gt
                    maxDtIndex = [-1 for _ in range(len(gt))]  # store dt index with maximum IoF for each gt

                    for dind, d in enumerate(dt): 
                        # dt already matched
                        if dtm[tind, dind] > 0 or dtIg[tind, dind] > 0:
                            continue
                        #
                        m = -1 
                        for gind, g in zip(is_group_idx, group_gt): 
                            iof = iofs[dind, gind]
                            if iof > iof_thresh:  
                                if iof > maxIoF[gind]:  # if current IoF is larger than stored maxIoF
                                    if maxDtIndex[gind] != -1:  # if there was a previously stored dt for this gt
                                        dtIg[tind,maxDtIndex[gind]] = 1  # ignore the previous dt
                                        tmp = int(maxDtIndex[gind])
                                        
                                    maxIoF[gind] = iof
                                    maxDtIndex[gind] = dind
                                    dtIg[tind,dind] = 0  # not ignored
                                    m = gind 
                                else:
                                    dtIg[tind,dind] = 1  # ignore other dts inside gt 

                        if m == -1:
                            continue 

                        # if match made store id of match for both dt and gt
                        dtm[tind,dind]  = gt[m]['id']
                        gtm[tind,m]     = d['id']

        else: # normal 
            if not len(ious)==0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t,1-1e-10])
                        m   = -1
                        for gind, g in enumerate(gt):
                            # if this gt already matched, and not a crowd, continue
                            if gtm[tind,gind]>0 and not iscrowd[gind]:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                                break

                            # continue to next gt unless better match made
                            if ious[dind,gind] < iou:
                                continue
                            # if match successful and best so far, store appropriately
                            iou=ious[dind,gind]
                            m=gind

                        if m ==-1:
                            continue

                        # if match made store id of match for both dt and gt
                        dtIg[tind,dind] = gtIg[m] 
                        dtm[tind,dind]  = gt[m]['id']
                        gtm[tind,m]     = d['id']



        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))

        for gind, g in enumerate(gt):
            if gtIg[gind] != 1:
                gid = g['category_id']
                if gid not in self.gt_dt_valid:
                    self.gt_dt_valid[gid] = {'gts': 0, 'dts': [0]*T}
                self.gt_dt_valid[gid]['gts'] += 1

        for dind, d in enumerate(dt):
            for tind in range(len(p.iouThrs)):
                if dtIg[tind, dind] != 1:
                    did = d['category_id']
                    if did not in self.gt_dt_valid:
                        self.gt_dt_valid[did] = {'gts': 0, 'dts': [0]*T}
                    self.gt_dt_valid[did]['dts'][tind] += 1

        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
              }
    
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])

            if get_rank() == 0:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

            return mean_s

        def _summarizeDets():
            stats = np.zeros((13,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            # add 
            stats[12] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
