# fmri_analysis_core.py

from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import h5py
import cortex
from collections import defaultdict
import nibabel as nib
from typing import Dict, List, Optional


# ---------- CONFIG ----------

stories = [
    "wheretheressmoke",
    "fromboyhoodtofatherhood",
    "onapproachtopluto",
]

subject_xfm = {
    'AA': '20180905AA-sg-auto',
    'AHfs': '20180816AH-sg',
    'S1': '20180414SJ-sg_auto3',
    'S4': '20190121SS_auto',
    'S5': '20190715_sg_auto',
    'S6': '20190910IG_sg_auto',
    'S7': '20190916RA_auto',
    'BFD001': '20220421BFD001-sg-auto',
    'BFD003': '20230730BFD003-sg-auto'
}

base_dir = Path("/Users/jiamingxu/Desktop/Desktop - Jiaming’s MacBook Pro/Language_straightening/data/language")
ROI_DIR_LANGUAGE = Path("/Users/jiamingxu/Desktop/Desktop - Jiaming’s MacBook Pro/Language_straightening/data/rois")
ROI_DIR_VISION = Path("/Users/jiamingxu/Desktop/Desktop - Jiaming’s MacBook Pro/Language_straightening/data/vision_fixation")
freesurfer_dir = Path("/Users/jiamingxu/Desktop/Desktop - Jiaming’s MacBook Pro/Language_straightening/data/freesurfer")
DB = cortex.db  


# ---------- LOW-LEVEL HELPERS ----------

def hf5_path(subject: str, story: str) -> Path:
    return base_dir / f"{story}" / f"{subject}_{story}.hf5"


@lru_cache(maxsize=None)
def load_response_trials(subject: str, story: str, drop_first_trs: int = 50) -> np.ndarray:
    path = hf5_path(subject, story)
    with h5py.File(path, "r") as f:
        dsname = next(iter(f.keys()))
        arr = f[dsname][:]
    if drop_first_trs:
        arr = arr[:, drop_first_trs:, :]
    return arr


@lru_cache(maxsize=None)
def load_mask(subject: str) -> np.ndarray:
    xfm = subject_xfm[subject]
    return DB.get_mask(subject, xfm)

# for getting A1 voxels
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
    # print("making single hem to full surface...")

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


def get_xfm(subject: str) -> str:
    return subject_xfm[subject]


# ---------- FUNCTIONS ----------

def build_roi_voxs(
    subject: str,
    f_rois=None, # default is AC and sPMv
    use_language_rois: bool = True,
) -> dict:
    """
    Build a dict roi_voxs[roi_name] -> voxel indices (1D indices into masked voxels)
    combining functional and anatomical ROIs.
    """
    if f_rois is None:
        f_rois = ['AC', 'sPMv']

    # ---------- functional ROIs ----------
    xfm = get_xfm(subject)
    mask = load_mask(subject)   # 3D brain mask

    roi_masks = cortex.utils.get_roi_masks(
        subject,
        xfm,
        roi_list=f_rois,
        gm_sampler='cortical',
        split_lr=False,
        threshold=None,
        return_dict=True
    )

    f_roi_voxs = {}
    for roi in f_rois:
        roi_mask = roi_masks[roi]                  # 3D mask for this ROI
        f_roi_voxs[roi] = np.where(roi_mask[np.where(mask)])[0]

    # ---------- anatomical ROIs ----------
    if use_language_rois:
        roi_dir = ROI_DIR_LANGUAGE
    else:
        roi_dir = ROI_DIR_VISION

    a_roi = [
        'parsopercularis', 'parstriangularis', 'superiorfrontal',
        'rostralmiddlefrontal','caudalmiddlefrontal','frontalpole',
        'precuneus'
    ]

    roi_path = roi_dir / f"{subject}_roi.npy"
    roi_data = np.load(roi_path, allow_pickle=True).item()

    a_roi_voxs = {roi: roi_data[roi] for roi in a_roi}

    # combine PFC rois
    rois_to_combine = [
        'parsopercularis',
        'parstriangularis',
        'superiorfrontal',
        'rostralmiddlefrontal',
        'caudalmiddlefrontal',
        'frontalpole'
    ]

    pfc_voxs = []
    for roi in rois_to_combine:
        pfc_voxs.extend(a_roi_voxs[roi])
    pfc_voxs = list(set(pfc_voxs))  # remove potential duplicates

    a_roi_voxs['prefrontal'] = pfc_voxs
    for roi in rois_to_combine:
        del a_roi_voxs[roi]

    # ---------- merge functional + anatomical ----------
    roi_voxs = {**f_roi_voxs, **a_roi_voxs}

    return roi_voxs


def build_A1_voxs(
    subjects: List[str],
    xfms: Optional[Dict[str, str]] = None,
    freesurferdir: Optional[str] = None,
    dkt_label_index: int = 34,
    verts2vox_threshold: float = 0.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build A1 (AC ∩ transverse temporal) and AC-nonA1 voxel indices for each subject.

    IMPORTANT: Output indices are in *cortex-mask order* (1D), so they can index
    response_trials[..., voxels] directly.

    Returns:
        out[subject] = {
            "A1_L": 1D int indices,
            "A1_R": 1D int indices,
            "A1_all": 1D int indices,
            "nonA1_L": 1D int indices,
            "nonA1_R": 1D int indices,
            "nonA1_all": 1D int indices,
        }
    """
    if xfms is None:
        xfms = subject_xfm
    if freesurferdir is None:
        freesurferdir = freesurfer_dir

    # 1) Load DKT annotations (per subject, per hemisphere)
    dkts = defaultdict(list)  # {subject: [lh_annot, rh_annot]}
    for subject in subjects:
        for hem in ["lh", "rh"]:
            annot_path = f"{freesurferdir}/{subject}/label/{hem}.aparc.DKTatlas.annot"
            if subject == "AHfs":
                annot_path = f"{freesurferdir}/{subject}/label/{hem}.aparc.DKTatlas40.annot"

            annot_data, ctab, names = nib.freesurfer.io.read_annot(annot_path)
            dkts[subject].append(annot_data)

    # transverse temporal vertices in DKT (label index)
    transtemp_verts = {
        subject: [(hemdkt == dkt_label_index) for hemdkt in subjdkts]  # [lh_bool, rh_bool]
        for subject, subjdkts in dkts.items()
    }

    # 2) Map transverse temporal verts -> voxel masks (volume space)
    transtemp_voxs_3d = {}
    for subject, (lh_bool, rh_bool) in transtemp_verts.items():
        dd = {"transversetemporal_lh": lh_bool, "transversetemporal_rh": rh_bool}
        fullverts = vertex_hem2full(dd, subject)  # must exist in your codebase
        v2v_vol = verts_to_voxels(
        subject, xfms[subject], fullverts,
        threshold=None,
        ret_shape="volume",   # <-- so it's 3D volume masks
        # v2v = verts_to_voxels(subject, xfms[subject], fullverts, threshold=0
    )
    transtemp_voxs_3d[subject] = {k: (v > verts2vox_threshold) for k, v in v2v_vol.items()}
    # transtemp_voxs_3d[subject] = {k: (v > verts2vox_threshold) for k, v in v2v.items()}
    
    # 3) Load AC masks in volume space
    roi_masks_vols = {}
    for subject in subjects:
        roi_masks_vols[subject] = cortex.utils.get_roi_masks(
            subject, xfms[subject],
            roi_list=["AC"],
            gm_sampler="cortical",
            split_lr=True,
            threshold=None,
            return_dict=True,
            allow_overlap=False,
        )

    # 4) Convert everything into cortex-mask order (1D)
    cortex_masks_3d = {
        subject: cortex.db.get_mask(subject, xfms[subject], type="thick")
        for subject in subjects
    }

    # AC masks -> 1D
    ac_1d = {
        subject: {rn: m[cortex_masks_3d[subject]].astype(bool) for rn, m in roi_masks_vols[subject].items()}
        for subject in subjects
    }

    # transverse temporal masks -> 1D
    tt_1d = {
        subject: {
            "transversetemporal_lh": transtemp_voxs_3d[subject]["transversetemporal_lh"][cortex_masks_3d[subject]].astype(bool),
            "transversetemporal_rh": transtemp_voxs_3d[subject]["transversetemporal_rh"][cortex_masks_3d[subject]].astype(bool),
        }
        for subject in subjects
    }

    # 5) Intersect AC with transverse temporal => A1; and compute AC-nonA1
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for subject in subjects:
        AC_L = ac_1d[subject]["AC_L"]
        AC_R = ac_1d[subject]["AC_R"]

        TT_L = tt_1d[subject]["transversetemporal_lh"]
        TT_R = tt_1d[subject]["transversetemporal_rh"]

        A1_L_mask = AC_L & TT_L
        A1_R_mask = AC_R & TT_R

        nonA1_L_mask = AC_L & ~A1_L_mask
        nonA1_R_mask = AC_R & ~A1_R_mask

        out[subject] = {
            "A1_L": np.flatnonzero(A1_L_mask),
            "A1_R": np.flatnonzero(A1_R_mask),
            "A1_all": np.unique(np.concatenate([np.flatnonzero(A1_L_mask), np.flatnonzero(A1_R_mask)])),
            "nonA1_L": np.flatnonzero(nonA1_L_mask),
            "nonA1_R": np.flatnonzero(nonA1_R_mask),
            "nonA1_all": np.unique(np.concatenate([np.flatnonzero(nonA1_L_mask), np.flatnonzero(nonA1_R_mask)])),
        }

    return out