"""Forecast summary coordinates from a real prefix; never consume a known future."""
import argparse,json
import torch
from .session import ForecastSession

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',required=True);p.add_argument('--head',required=True)
    p.add_argument('--text',default='The new research report describes ');p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(1);s=ForecastSession(a.teacher,a.head,a.device)
    ids=torch.tensor([list(a.text.encode('utf-8'))],dtype=torch.long,device=a.device)
    s.observe(ids);f=s.forecast()
    means=f.mean.reshape(len(s.head.config.horizons),s.head.config.variables)
    record={'status':'forecast_only_not_committed','teacher_sha256':s.teacher_hash,'head_sha256':s.head_hash,
        'observed_bytes':len(ids[0]),'horizons_in_bytes':s.head.config.horizons,'standardized_summary_means':means.tolist(),
        'mixture_probabilities':f.logits.softmax(-1).tolist()[0],
        'point_forecast_status':s.point_forecast_status(),
        'warning':'These are eight lossy memory-summary quantities, not text, full P/S states, or verified future facts.'}
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
