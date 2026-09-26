"""Opt-in forecast-to-logit residual with a separately frozen target teacher."""
import torch
from torch import nn
from energy_real.summaries import forward_with_summaries,energy_layer
from energy_forecast.model import ForecastConfig,ForecastEnergy


class JointForecastLM(nn.Module):
    def __init__(self,backbone,center,scale,*,layer=None,forecast_weight=.1):
        super().__init__();self.backbone=backbone;self.layer=energy_layer(backbone.config,layer);self.forecast_weight=forecast_weight
        variables=2*backbone.config.heads
        if center.shape!=(variables,) or scale.shape!=(variables,) or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid frozen normalization')
        self.register_buffer('center',center.clone());self.register_buffer('scale',scale.clone())
        self.forecast=ForecastEnergy(ForecastConfig(variables=variables))
        self.output=nn.Linear(self.forecast.d,backbone.config.vocab,bias=False);nn.init.zeros_(self.output.weight)

    def forward(self,tokens,*,future_targets=None):
        logits,state,summary=forward_with_summaries(self.backbone,tokens,layer=self.layer)
        # Inputs only contain real observed prefixes. Targets enter the loss below.
        history=(summary-self.center)/self.scale
        padded=torch.nn.functional.pad(history,(0,0,15,0))
        windows=padded.unfold(1,16,1).transpose(-1,-2)
        b,t,_,v=windows.shape;density=self.forecast(windows.reshape(b*t,16,v))
        result=logits+self.output(density.mean).reshape(b,t,-1)
        auxiliary=None
        if future_targets is not None:
            if future_targets.requires_grad:raise ValueError('Frozen targets must be detached')
            if future_targets.shape!=(b,t,4,v):raise ValueError('Wrong teacher targets')
            valid=torch.isfinite(future_targets).all(dim=(-1,-2));valid[:,:15]=False
            safe=torch.nan_to_num(future_targets).reshape(b*t,-1)
            losses=density.energy(safe).reshape(b,t)/self.forecast.d
            auxiliary=losses[valid].mean() if valid.any() else losses.sum()*0
        return result,state,auxiliary


@torch.no_grad()
def frozen_targets(teacher,tokens,center,scale,layer=None):
    teacher.eval();_,_,summary=forward_with_summaries(teacher,tokens,layer=layer)
    summary=(summary-center)/scale;b,t,v=summary.shape
    targets=summary.new_full((b,t,4,v),float('nan'))
    for i,h in enumerate((1,4,16,64)):
        if t>h:targets[:,:t-h,i]=summary[:,h:]
    return targets
