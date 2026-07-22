"""Expanded real-pixel CT restoration proxy.

Uses six CT frames from five patient IDs in the public pydicom/pydicom-data
package. Evaluation is leave-one-patient-out. For every held-out patient/frame,
six independently seeded low-dose-like corruption cases are evaluated.
This is NOT the AI-BOOST missing-acquisition ICR KPI and NOT clinical validation.
"""
from __future__ import annotations
import csv, hashlib, json, os, random
from pathlib import Path
from typing import Dict, List, Tuple
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import cv2, numpy as np, pydicom, torch, torch.nn as nn
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import t
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle
from torch.utils.data import DataLoader, Dataset
SEED=20260808
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))
ROOT=Path(__file__).resolve().parent
OUT_JSON=ROOT/'results_real_ct_pixel.json'; OUT_CSV=ROOT/'results_real_ct_pixel_cases.csv'; OUT_MODEL=ROOT/'real_ct_residual_cnn.pt'

def locate_files():
 import data_store, pydicom as pd
 dr=Path(data_store.__file__).resolve().parent/'data'; pr=Path(pd.__file__).resolve().parent/'data'/'test_files'
 return [dr/'693_UNCI.dcm',pr/'J2K_pixelrep_mismatch.dcm',pr/'CT_small.dcm',dr/'eCT_Supplemental.dcm',dr/'explicit_VR-UN.dcm']

def read_cases():
 cases=[]
 for p in locate_files():
  ds=pydicom.dcmread(str(p),force=True)
  if getattr(ds,'Modality',None)!='CT': continue
  arr=ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1.0))+float(getattr(ds,'RescaleIntercept',0.0))
  frames=[arr] if arr.ndim==2 else [arr[i] for i in range(arr.shape[0])]
  pid=str(getattr(ds,'PatientID',p.stem)); study=str(getattr(ds,'StudyInstanceUID',''))
  for i,im in enumerate(frames): cases.append({'patient_id':pid,'study_uid':study,'source':p,'frame':i,'image':resize_norm(im)})
 return cases

def resize_norm(im,size=512):
 finite=im[np.isfinite(im)]; lo,hi=np.percentile(finite,[0.5,99.5]); hi=max(hi,lo+1)
 x=np.clip((im-lo)/(hi-lo),0,1).astype(np.float32)
 return x if x.shape==(size,size) else cv2.resize(x,(size,size),interpolation=cv2.INTER_CUBIC).astype(np.float32)

def corrupt(im,rng,level):
 configs=[(90,.012),(70,.018),(50,.024)]
 photons,sigma=configs[level]
 noisy=rng.poisson(np.clip(im,0,1)*photons).astype(np.float32)/photons
 noisy+=rng.normal(0,sigma,im.shape).astype(np.float32)
 return np.clip(noisy,0,1)

class PatchDataset(Dataset):
 def __init__(self,ims,n=360,patch=64): self.ims=ims; self.n=n; self.patch=patch
 def __len__(self): return self.n
 def __getitem__(self,idx):
  rng=np.random.default_rng(SEED+idx*7919); im=self.ims[int(rng.integers(0,len(self.ims)))]; h,w=im.shape
  for _ in range(10):
   y=int(rng.integers(0,h-self.patch+1)); x=int(rng.integers(0,w-self.patch+1)); clean=im[y:y+self.patch,x:x+self.patch]
   if clean.std()>.035: break
  noisy=corrupt(clean,rng,int(rng.integers(0,3)))
  return torch.from_numpy(noisy[None]),torch.from_numpy(clean[None])
class ResidualCNN(nn.Module):
 def __init__(self,c=12):
  super().__init__(); layers=[nn.Conv2d(1,c,3,padding=1),nn.ReLU()]
  for _ in range(3): layers += [nn.Conv2d(c,c,3,padding=1),nn.ReLU()]
  layers += [nn.Conv2d(c,1,3,padding=1)]; self.net=nn.Sequential(*layers)
 def forward(self,x): return x-self.net(x)
def train(ims,fold):
 torch.manual_seed(SEED+fold); model=ResidualCNN(); dl=DataLoader(PatchDataset(ims),batch_size=48,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED+fold))
 opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss=nn.L1Loss(); hist=[]
 for _ in range(2):
  s=n=0
  for noisy,clean in dl:
   opt.zero_grad(); pred=model(noisy); l=loss(pred,clean); l.backward(); opt.step(); s+=l.item()*len(noisy); n+=len(noisy)
  hist.append(s/n)
 return model.eval(),hist

def ccc(x,y):
 x=np.asarray(x,float); y=np.asarray(y,float); den=x.var()+y.var()+(x.mean()-y.mean())**2
 return float(2*np.mean((x-x.mean())*(y-y.mean()))/den) if den>0 else 1.0

def texture(im,patch=64,levels=32):
 feats={k:[] for k in ['entropy','contrast','homogeneity','energy']}
 for y in range(0,im.shape[0]-patch+1,patch):
  for x in range(0,im.shape[1]-patch+1,patch):
   p=im[y:y+patch,x:x+patch]; q=np.clip(np.floor(p*(levels-1)),0,levels-1).astype(np.uint8)
   g=graycomatrix(q,[1],[0],levels=levels,symmetric=True,normed=True); prob=g[:,:,0,0]; nz=prob[prob>0]
   feats['entropy'].append(float(-(nz*np.log2(nz)).sum()))
   for k in ['contrast','homogeneity','energy']: feats[k].append(float(graycoprops(g,k)[0,0]))
 return {k:np.asarray(v) for k,v in feats.items()}

def metrics(clean,pred):
 pred=np.clip(pred,0,1); ec=np.hypot(sobel(clean,0),sobel(clean,1)); ep=np.hypot(sobel(pred,0),sobel(pred,1))
 mc=clean>.08; mp=pred>.08; dice=2*np.logical_and(mc,mp).sum()/max(1,mc.sum()+mp.sum())
 edge_thr=np.percentile(ec,85); ce=ec>edge_thr; pe=ep>edge_thr
 halluc=np.logical_and(pe,~ce).sum()/max(1,pe.sum()); deleted=np.logical_and(ce,~pe).sum()/max(1,ce.sum())
 tc,tp=texture(clean),texture(pred)
 return {'mae':float(np.mean(np.abs(clean-pred))),'psnr_db':float(peak_signal_noise_ratio(clean,pred,data_range=1)),'ssim':float(structural_similarity(clean,pred,data_range=1)),'edge_correlation':float(np.corrcoef(ec.ravel(),ep.ravel())[0,1]),'foreground_dice_proxy':float(dice),'hallucinated_edge_fraction_proxy':float(halluc),'deleted_edge_fraction_proxy':float(deleted),**{f'glcm_{k}_ccc':ccc(tc[k],tp[k]) for k in tc}}

def summary(vals):
 a=np.asarray(vals,float); n=len(a); mean=float(a.mean()); sd=float(a.std(ddof=1)) if n>1 else 0.; half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else 0.
 return {'n':n,'mean':mean,'sd':sd,'ci95_low':mean-half,'ci95_high':mean+half}
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
 cases=read_cases(); pids=sorted({c['patient_id'] for c in cases}); rows=[]; fold_hist={}
 for fi,pid in enumerate(pids):
  trainims=[c['image'] for c in cases if c['patient_id']!=pid]; model,hist=train(trainims,fi); fold_hist[pid]=hist
  for c in [x for x in cases if x['patient_id']==pid]:
   clean=c['image']
   for level in range(3):
    for rep in range(2):
     rng=np.random.default_rng(SEED+fi*1000+c['frame']*100+level*10+rep); noisy=corrupt(clean,rng,level)
     preds={'noisy':noisy,'gaussian':np.clip(gaussian_filter(noisy,.8),0,1),'tv':np.clip(denoise_tv_chambolle(noisy,weight=.10,channel_axis=None),0,1).astype(np.float32)}
     with torch.no_grad(): preds['residual_cnn']=np.clip(model(torch.from_numpy(noisy[None,None])).squeeze().numpy(),0,1)
     for method,pred in preds.items(): rows.append({'patient_id':pid,'source':c['source'].name,'frame':c['frame'],'corruption_level':['mild','moderate','severe'][level],'repeat':rep,'method':method,**metrics(clean,pred)})
 # save per-case
 fields=list(rows[0]);
 with OUT_CSV.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 numeric=[k for k in rows[0] if k not in {'patient_id','source','frame','corruption_level','repeat','method'}]
 summaries={m:{k:summary([r[k] for r in rows if r['method']==m]) for k in numeric} for m in ['noisy','gaussian','tv','residual_cnn']}
 best=max(['gaussian','tv','residual_cnn'],key=lambda m:summaries[m]['ssim']['mean'])
 res={'evidence_scope':'Patient-held-out real medical CT pixel restoration proxy with synthetic low-dose-like corruption; not official ICR and not clinical validation.','dataset_provenance':'Public CT DICOM test data distributed with the MIT-licensed pydicom/pydicom-data packages.','dicom_files':len(locate_files()),'unique_patient_ids':len(pids),'ct_frames':len(cases),'split':'Five-fold leave-one-patient-ID-out; no held-out patient pixels used for CNN training.','corruption_protocol':'Three Poisson-Gaussian severity levels, two independent repetitions per frame.','evaluations_per_method':len([r for r in rows if r['method']=='noisy']),'methods':['corrupted input','Gaussian filter','TV denoising','small residual CNN'],'best_method_by_mean_ssim':best,'summary':summaries,'fold_training_loss':fold_hist,'case_level_csv':OUT_CSV.name,'source_files':[{'name':p.name,'sha256':sha(p)} for p in locate_files()]}
 OUT_JSON.write_text(json.dumps(res,indent=2),encoding='utf-8')
 print(json.dumps({'patients':len(pids),'frames':len(cases),'cases_per_method':res['evaluations_per_method'],'best':best,'best_ssim':summaries[best]['ssim']},indent=2))
if __name__=='__main__': main()
