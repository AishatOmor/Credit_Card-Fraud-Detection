from pathlib import Path
import pandas as pd,numpy as np,json,shutil,zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,confusion_matrix
import nbformat
root=Path(__file__).resolve().parents[1];(root/'outputs').mkdir(parents=True,exist_ok=True);(root/'docs').mkdir(exist_ok=True);(root/'notebooks').mkdir(exist_ok=True);(root/'dashboard').mkdir(exist_ok=True)
source=root/'creditcard.csv'
df=pd.read_csv(source).sort_values('Time',kind='stable').reset_index(drop=True)
features=[c for c in df if c!='Class']

def fit_evaluate(frame,mode):
 a=int(len(frame)*.6);b=int(len(frame)*.8);tr,va,te=frame.iloc[:a],frame.iloc[a:b],frame.iloc[b:]
 scaler=StandardScaler();xt=scaler.fit_transform(tr[features]);xv=scaler.transform(va[features]);xe=scaler.transform(te[features]);yv=va.Class.to_numpy();ye=te.Class.to_numpy()
 results=[];scores={}
 for name,model in [('Logistic regression',LogisticRegression(class_weight='balanced',max_iter=300,random_state=42)),('Hist gradient boosting',HistGradientBoostingClassifier(max_iter=110,max_depth=4,learning_rate=.08,l2_regularization=1,random_state=42))]:
  model.fit(xt,tr.Class);vs=model.predict_proba(xv)[:,1];s=model.predict_proba(xe)[:,1]
  p,r,t=precision_recall_curve(yv,vs);eligible=np.flatnonzero(r[:-1]>=.8);idx=int(eligible[np.argmax(p[:-1][eligible])]);threshold=float(t[idx]);pred=s>=threshold
  tn,fp,fn,tp=confusion_matrix(ye,pred,labels=[0,1]).ravel()
  results.append(dict(experiment=mode,model=name,n_test=len(ye),fraud_test=int(ye.sum()),roc_auc=roc_auc_score(ye,s),pr_auc=average_precision_score(ye,s),precision=tp/(tp+fp),recall=tp/(tp+fn),false_positive_rate=fp/(fp+tn),true_positive=int(tp),false_positive=int(fp),false_negative=int(fn),true_negative=int(tn),validation_threshold=threshold))
  scores[name]=s
  if mode=='original':
   coefs=pd.DataFrame({'feature':features,'standardized_coefficient':model.coef_[0]}) if name=='Logistic regression' else None
   if coefs is not None:
    coefs['absolute_coefficient']=coefs.standardized_coefficient.abs();coefs.sort_values('absolute_coefficient',ascending=False).to_csv(root/'outputs'/'logistic_feature_coefficients.csv',index=False)
 if mode=='original':
  rng=np.random.default_rng(20261118);boots=[]; n=len(ye)
  for i in range(400):
   ix=rng.integers(0,n,n);y=ye[ix]
   if y.min()==y.max():continue
   row={'iteration':i}
   for name in scores:
    row[name+'_roc_auc']=roc_auc_score(y,scores[name][ix]);row[name+'_pr_auc']=average_precision_score(y,scores[name][ix])
   row['delta_pr_auc_logistic_minus_boosting']=row['Logistic regression_pr_auc']-row['Hist gradient boosting_pr_auc'];boots.append(row)
  bs=pd.DataFrame(boots); intervals=[]
  for col in [c for c in bs if c!='iteration']:
   intervals.append({'metric':col,'lower_95':float(bs[col].quantile(.025)),'upper_95':float(bs[col].quantile(.975)),'bootstrap_samples':len(bs)})
  pd.DataFrame(intervals).to_csv(root/'outputs'/'bootstrap_intervals.csv',index=False)
 return results
rows=fit_evaluate(df,'original')
# Sensitivity removes exact duplicate rows before chronological split; not directly matched test populations.
dedup=df.drop_duplicates().reset_index(drop=True)
rows+=fit_evaluate(dedup,'deduplicated')
metrics=pd.DataFrame(rows);metrics.to_csv(root/'outputs'/'duplicate_sensitivity_metrics.csv',index=False)
# Duplicate overlap audit across original chronological splits
n=len(df);a=int(n*.6);b=int(n*.8)
keys=pd.util.hash_pandas_object(df, index=False).to_numpy();train=set(keys[:a]);val=set(keys[a:b]);test=set(keys[b:]);cross={'train_validation_overlap_unique_row_hashes':len(train&val),'train_test_overlap_unique_row_hashes':len(train&test),'validation_test_overlap_unique_row_hashes':len(val&test),'duplicates_removed':int(len(df)-len(dedup))}
(root/'outputs'/'duplicate_overlap_audit.json').write_text(json.dumps(cross,indent=2))
print(metrics[['experiment','model','roc_auc','pr_auc','precision','recall','false_positive']].to_string(index=False))
