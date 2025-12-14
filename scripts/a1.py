'''
2025.08.06 Suna Guo
Get vox/verts for A1 from DKT(40) atlas transversetemporal ROI

Before using, make sure you have the freesurfer dir for the subject, 
and they have `{subject}/label/{hem}.aparc.DKTatlas<40>.annot` 
(contain "40" in older version of fs dir).
'''

import numpy as np
import cortex
## ======================================
## ========== helper functions ==========
## ======================================
def verts_to_voxels(subject, xfm, verts, sep_roi=True, 
                    mapper_sampler="nearest", gm_sampler="thick", ret_shape="cortex", 
                    threshold=None
                   ):
    """
    Map vertices to voxels with mapper.backwards.

    Input:
    verts: {roi_name: [vert inds in the ROI]} or {roi_name: [full verts bool of roi verts]}
    ret_shape: [default] "cortex" - 1D array of cortex map shape;
               "volume": 3D array of (54, 84, 84).
    Output
    rois_voxels: {roi_name: vox vals}. Note func doesn't return bools even when threshold is provided, 
                 to ensure returned data are compatible

    TODO: update to fancy version here: https://github.com/gallantlab/pycortex/blob/main/cortex/utils.py#L476
    such that pts removed from surf cuts are included in the result list
    """
    
    mapper = cortex.get_mapper(subject, xfm, mapper_sampler, recache=False)
    
    ## bool volume of which vox is included
    rois_voxels = {}
    for rname, rvertsinds in verts.items():
        if len(rvertsinds) == mapper.nverts: 
            rois_voxels[rname] = mapper.backwards(rvertsinds)
        elif len(rvertsinds) < mapper.nverts: 
            fullrverts = np.zeros(mapper.nverts)
            fullrverts[rvertsinds] = 1
            rois_voxels[rname] = mapper.backwards(fullrverts)
        else:
            raise ValueError(f"nverts doesn't match: input inds/verts {len(rvertsinds)}, mapper {mapper.nverts}")
        
    if threshold is not None: 
        rois_voxels = {rn: (v > threshold).astype(int) for rn, v in rois_voxels.items()}

    if ret_shape == "cortex":
        cortex_mask = cortex.db.get_mask(subject, xfm, type=gm_sampler)
        ret = {rn: v[cortex_mask] for rn, v in rois_voxels.items()}
    ## TODO: fix param combo of ret_shape="volume" & sep_roi=True
    elif ret_shape == "volume":
        ret = rois_voxels
    else: 
        raise ValueError(f"return shape {ret_shape} not reccognized! ([cortex]/volume)")

    ## put all ROIs in single volume
    if not sep_roi:
        ret = np.any(list(rois_voxels.values()), axis=0)
    
    return ret

def vertex_hem2full(data, subject, empty=0):
    """
    Make full vertex array (with both hem) with given hem data.
    Follow dict key format for funct to identify left/right hem: 
    e.g. use <ROI>_l or <ROI>_lh 
    Must follow for fsaverage (left/right hem have same number of verts), 
    but for usual subjects we can also identify by nverts from each hem

    Input:
    data: dict {<ROI>_l: [lh verts], <ROI>_r: [rh verts]}
    empty: 0 or nan

    Output: 
    eps: dict {<ROI>_l: [full verts], <ROI>_r: [full verts]}
         (use same keys as they came in)
    """
    
    ## get number of vertices for each hem 
    ## source: pycortex gallery
    print("making single hem to full surface...")

    surfs = [cortex.polyutils.Surface(*d)
            for d in cortex.db.get_surf(subject, "fiducial")]
    num_verts = np.array([s.pts.shape[0] for s in surfs])

    eps = {}
    for tn, ep in data.items():
        hem = tn.split("_")[-1][0].lower()
        if hem not in ["l", "r"]:
            hem = np.array(["l", "r"])[ep.shape==num_verts]
                
        ## vertex: lh then rh
        if "l" in hem:
            lh = ep
            rh = np.zeros(num_verts[1]) * empty
        elif "r" in hem:
            lh = np.zeros(num_verts[0]) * empty
            rh = ep

        eps[tn] = np.concatenate([lh, rh])
    
    return eps

## ==================================
## ========== main: get A1 ==========
## ==================================
from collections import defaultdict
import nibabel as nib

## fill in these vars
freesurferdir = ""
subjects = []

## load DKT(40) atlas from their freesurfer labels
## {subject: [lh_verts, rh_verts]}
dkts = defaultdict(list)
for subject in subjects: 
    for hem in ["lh", "rh"]:
        annot_path = f'{freesurferdir}/{subject}/label/{hem}.aparc.DKTatlas.annot'
        if subject == "AHfs": 
            annot_path = f'{freesurferdir}/{subject}/label/{hem}.aparc.DKTatlas40.annot'
        
        ## annot_data: list of nverts for hem, contains atlas ROI indices 
        ## ctab: color table, not used here
        ## names: list of atlas ROI names 
        ## * note that a hem may not contain all ROIs, 
        ## * so the number of unique values in annot_data may not match total number of names
        annot_data, ctab, names = nib.freesurfer.io.read_annot(annot_path)
            
        dkts[subject].append(annot_data)

## transverse temporal is labeled 34 in DKT atlas (names[34])
transtemp_verts = {subject: [d == 34 for d in subjdkts] for subject, subjdkts in dkts.items()}

## =====================================================================================
## ===== stop here if you just need the vertices for full transversetemporal as A1 =====
## =====================================================================================

import cottoncandy as cc
cci = cc.get_interface("story-mri-data", verbose=False)
xfms = cci.download_json("subject_xfms")

## verts2vox interpolates data, so binary masks becomes continuous values
## this is arbitrary, setting to 0 includes all vox projected to by any verts (most loose)
verts2vox_threshold = 0

## mapping verts to voxels
## {subject: {transversetemporal_lh: [full vox mask], transversetemporal_rh: [full vox mask]}}
transtemp_voxs = {}
for subject, (lh, rh) in transtemp_verts.items():
    dd = {"transversetemporal_lh": lh, "transversetemporal_rh": rh}
    ## verts to vox only work on full cortex array
    fullverts = vertex_hem2full(dd, subject)
    transtemp_voxs[subject] = verts_to_voxels(subject, xfms[subject], fullverts, threshold=0)
    transtemp_voxs[subject] = {k: v > verts2vox_threshold for k, v in transtemp_voxs[subject].items()}

## ===================================================================================
## ===== stop here if you just need the voxels for full transversetemporal as A1 =====
## ===================================================================================

## load vox for AC to intersect with transtemp
## {subject: {"AC_L": [3D volume mask], "AC_R": [3D volume mask]}}
roi_masks_vols = {}
for subject in subjects:
    roi_masks_vols[subject] = cortex.utils.get_roi_masks(subject, xfms[subject],
                                roi_list=["AC"], # Default (None) gives all available ROIs in overlays.svg
                                gm_sampler='thick', # Select only voxels mostly within cortex
                                split_lr=True, # Separate left/right ROIs (this occurs anyway with index volumes)
                                threshold=0, # convert probability values to boolean mask for each ROI
                                return_dict=True, # return index volume, or dict of masks
                                allow_overlap=True,  # otherwise overlapping voxels will be assigned to the first ROI in the list
                            )
## get mask that chooses cortex vox from 3D volume
masks = {subject: cortex.db.get_mask(subject, xfms[subject], type="thick")
         for subject in subjects}
## get ROI masks from 3D volume
roi_masks = {subject: {rn: m[masks[subject]] for rn, m in roi_masks_vols[subject].items()}
             for subject in roi_masks_vols.keys()}

## overlap btw AC & transtemp --> A1
## {subject: {"A1_L": [cortex mask], "A1_R": [cortex mask]}}
for subject, srmd in roi_masks.items():
    roi_masks[subject]["A1_L"] = srmd["AC_L"] & transtemp_voxs[subject]["transversetemporal_lh"]
    roi_masks[subject]["A1_R"] = srmd["AC_R"] & transtemp_voxs[subject]["transversetemporal_rh"]

## =======================================================================
## ===== now you have the intersect of transversetemporal & AC as A1 =====
## =======================================================================