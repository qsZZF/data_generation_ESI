"""Diagnostic plots of saved pilot results; no synthetic observations are invented."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('generated_simulation_dataset/G1_spectral_validation'))
    args=parser.parse_args();folder=args.directory
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','font.family':'DejaVu Sans'})
    measured='#555555';simulated='#277da8'
    rest=json.loads((folder/'rest_results.json').read_text())
    states=json.loads((folder/'state_results.json').read_text())
    with np.load(folder/'rest_plot_data.npz') as data, np.load(folder/'state_plot_data.npz') as state_data:
        fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
        f=data['frequency'];keep=(f>=2)&(f<=35)
        axes[0,0].plot(f[keep],10*np.log10(data['measured_test'][keep]),color=measured,label='EEGMAT held-out subjects')
        axes[0,0].plot(f[keep],10*np.log10(data['simulated'][keep]),color=simulated,label='Selected LBO; new seeds')
        axes[0,0].set(title='A  Resting spectral shape',xlabel='Frequency (Hz)',ylabel='Normalized PSD (dB/Hz)')
        axes[0,0].legend(fontsize=8)
        for ax,condition,title in [(axes[0,1],'eyes_open','B  Eyes open'),(axes[1,0],'eyes_closed','C  Eyes closed')]:
            ax.plot(f[keep],10*np.log10(state_data[condition+'_real'][keep]),color=measured,label='PhysioNet held-out')
            ax.plot(f[keep],10*np.log10(state_data[condition+'_sim'][keep]),color=simulated,label='State-specific simulation')
            ax.set(title=title,xlabel='Frequency (Hz)',ylabel='Normalized PSD (dB/Hz)');ax.legend(fontsize=8)
        lags=[25,50,100,250]
        axes[1,1].plot(lags,[rest['real_group_features']['test'][f'acf{x}'] for x in lags],'o-',color=measured,label='EEGMAT held-out median')
        axes[1,1].plot(lags,[rest['independent_seed_features'][f'acf{x}'] for x in lags],'o-',color=simulated,label='Selected LBO; new seeds')
        axes[1,1].axhline(0,color='.8',lw=.7)
        axes[1,1].set(title='D  Temporal autocorrelation',xlabel='Lag (ms)',ylabel='Autocorrelation');axes[1,1].legend(fontsize=8)
        fig.suptitle('G1 spectral pilot: held-out agreement varies by state; no universal realism claim',fontsize=11)
        save(fig,folder/'rest_validation')
    event=json.loads((folder/'event_results.json').read_text())
    erp=event['recommendations']['erp'];ersp=event['recommendations']['ersp']
    with np.load(folder/'event_plot_data.npz') as data:
        fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
        for split,marker,color in [('train','o','#b4b4b4'),('test','D',measured)]:
            rows=[row['features'] for row in event['real_reference'] if row['split']==split]
            axes[0,0].scatter([r['erp_peak_latency_s']*1000 for r in rows],
                             [r['erp_response_rms'] for r in rows],marker=marker,color=color,label='Real '+split)
        axes[0,0].scatter([erp['features']['erp_peak_latency_s']*1000],[erp['features']['erp_response_rms']],
                         marker='X',s=80,color=simulated,label='Selected simulation')
        axes[0,0].set(title='A  Cue average: 15 trials per point',xlabel='GFP peak latency (ms)',ylabel='ERP RMS / baseline RMS')
        axes[0,0].legend(fontsize=7)
        keys=['alpha_ersp_db','beta_ersp_db'];x=np.arange(2)
        for offset,source,label,color in [(-.25,ersp['real_train'],'Real train','#b4b4b4'),
                 (0,ersp['real_test'],'Real test',measured),(.25,ersp['features'],'Selected simulation',simulated)]:
            axes[0,1].bar(x+offset,[source[k] for k in keys],width=.24,label=label,color=color)
        axes[0,1].axhline(0,color='.6',lw=.7)
        axes[0,1].set(xticks=x,xticklabels=['Alpha (8-13 Hz)','Beta (13-30 Hz)'],
                      title='B  Motor-region ERSP; C3 / Cz / C4',ylabel='Task / prestimulus power (dB)')
        axes[0,1].legend(fontsize=7)
        freq=data['tf_frequency'];sel=(freq>=4)&(freq<=30)
        for ax,key,title in [(axes[1,0],'real_tf_test','C  Held-out motor imagery'),
                 (axes[1,1],f"tf_{ersp['candidate']}",'D  Selected induced-activity simulation')]:
            image=ax.pcolormesh(data['tf_time'],freq[sel],data[key][sel],shading='auto',cmap='RdBu_r',vmin=-4,vmax=4)
            ax.axvline(0,color='.2',ls=':',lw=.8)
            ax.set(title=title,xlabel='Time from cue (s)',ylabel='Frequency (Hz)')
        fig.colorbar(image,ax=axes[1,:].tolist(),label='Baseline-relative power (dB)',shrink=.8)
        fig.suptitle('Event pilot: temporal controls work; real ERP and ERSP remain imperfectly matched',fontsize=11)
        save(fig,folder/'event_validation')


if __name__=='__main__':main()
