"""Plot a compact comparison directly from the subject-independent evaluations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


def make_plot(root):
    def read(file):
        return json.loads((root/file).read_text(encoding='utf8'))

    for family in ('Microsoft YaHei','Noto Sans CJK SC','SimHei'):
        if family in font_manager.get_font_names():
            plt.rcParams['font.family'] = family
            break
    plt.rcParams['axes.unicode_minus'] = False
    stress = read('stress/mist_matched_results.json')['metrics']
    rest = read('stress/mist_pre_task_negative_control_results.json')['metrics']
    faced = read('emotion/evaluation.json')['results']['pooled_holdout']['metrics']
    met = read('emotion/met_transfer/results.json')['results']['metrics']
    focus = read('focus/results.json')['holdout']['scores']
    project = read('focus/verification.json')['project_six_matched_window']['holdout']['scores']

    def interval(name,obj,point,ci):
        return name,obj[point],obj[ci]

    panels = [
        ('压力实验条件 · MIST','受试者等权平衡准确率',[
            interval('心算，三频段',stress['relative']['macro_ba'],'value','ci95'),
            interval('仅任务前静息',rest['relative']['macro_ba'],'value','ci95'),
            interval('心算，当次静息校正',stress['calibrated']['macro_ba'],'value','ci95')]),
        ('情绪效价 · 视频和音乐','平衡准确率',[
            interval('FACED 视频，原权重',faced['old'],'ba','ba_ci'),
            interval('FACED 视频，个人历史',faced['history'],'ba','ba_ci'),
            interval('MET 音乐，冻结模型冷启动',met['history'],'ba','ba_ci')]),
        ('专注与分心 · Monash','受试者内平均 AUROC',[
            interval('仅行为',focus['behavior'],'macro_auc','ci95'),
            interval('行为＋项目原六项 EEG',project['project_six_quarter'],'macro_auc','ci95'),
            interval('行为＋优化双额极 EEG',focus['candidate'],'macro_auc','ci95')])]
    fig,axes = plt.subplots(3,1,figsize=(8.5,7.9))
    fig.subplots_adjust(left=.31,right=.91,top=.87,bottom=.14,hspace=1.02)
    for i,(ax,(title,xlabel,data)) in enumerate(zip(axes,panels)):
        ax.set_title(title,loc='left',fontweight='bold',fontsize=13,pad=7)
        lower = min(.35,min(bounds[0] for _,_,bounds in data)-.06)
        upper = max(.83,max(bounds[1] for _,_,bounds in data)+.07)
        ax.set_xlim(lower,upper)
        ax.axvline(.5,color='#817c72',linestyle='--',linewidth=1.1)
        for y,(label,point,(lo,hi)) in enumerate(data):
            color = '#006a68' if y==2 else '#3c5f84'
            ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color=color,capsize=3.5,linewidth=2)
            value = f'{point:.1%}' if i<2 else f'{point:.3f}'
            ax.text(hi+.006,y,value,ha='left',va='center',fontsize=10,color=color)
        ax.set_yticks(range(len(data)),[r[0] for r in data],fontsize=10)
        ax.set_ylim(len(data)-.55,-.55)
        ax.set_xlabel(xlabel,fontsize=10)
        if i<2:
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
        ax.grid(axis='x',alpha=.16)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#b3b3b3')
        ax.tick_params(axis='both',length=0)
    fig.text(.04,.986,'双额极脑电的受试者独立验证',fontsize=17,fontweight='bold',va='top')
    fig.text(.04,.035,'点为测试结果，横线为按受试者重采样的 95% 区间，虚线为随机参考。配对增量的区间见报告。',fontsize=9)
    fig.savefig(root/'双额极对比.png',dpi=180,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_root',type=Path)
    make_plot(parser.parse_args().result_root)
