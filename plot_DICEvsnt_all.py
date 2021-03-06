import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib


def get_dice(TESTSET_NAME, real_dataset_name, ax):
    DICE_dict = dict()
    nt_dict = dict()
    for model in ['proposed', 'resnet34', 'pulse_template_matching', 'cnn_slider']:
        if model != 'proposed':
            if model == 'resnet34':
                for exp in ['_CAM', '_SHAP']:
                    seg_report_folder = model + '/results/{}/segmentation_reports{}/'.format(TESTSET_NAME, exp)
                    if not os.path.exists(seg_report_folder): continue
                    DICE_dict[model + exp] = []
                    nt_dict[model + exp] = []
                    for f in os.listdir(seg_report_folder):
                        with open(seg_report_folder+f) as fp:
                            for i, line in enumerate(fp):
                                line = line.strip()
                                if i == 3:
                                    # line = float(line.split(' ')[0])
                                    DICE_dict[model+exp].append(line)
                            # line = float(line.split(' ')[0])
                            nt_dict[model+exp].append(line)
                        fp.close()
            else:
                seg_report_folder = model+'/results/{}/segmentation_reports/'.format(TESTSET_NAME)
                if not os.path.exists(seg_report_folder): continue
                DICE_dict[model] = []
                nt_dict[model] = []
                for f in os.listdir(seg_report_folder):
                    if len(f) == 17 and '10' not in f: continue
                    # print(f)

                    target_line = 3

                    with open(seg_report_folder+f) as fp:
                        for i, line in enumerate(fp):
                            line = line.strip()
                            if i == target_line:

                                if ']' in line:
                                    target_line = 4
                                    continue
                                # line = float(line.split(' ')[0])
                                DICE_dict[model].append(line)
                        # line = float(line.split(' ')[0])
                        nt_dict[model].append(line)
                    fp.close()
        else:
            seg_report_folder = model + '/results/{}/overall_eval_report.txt'.format(TESTSET_NAME)
            if not os.path.exists(seg_report_folder): continue
            DICE_dict[model] = []
            nt_dict[model] = []
            with open(seg_report_folder) as fp:
                for i, line in enumerate(fp):
                    line = line.strip()
                    if i == 3:
                        # line = float(line.split(' ')[0])
                        DICE_dict[model].append(line)
                # line = float(line.split(' ')[0])
                nt_dict[model].append(line)
            fp.close()

    x_loc = 0
    tick_pos = []
    colors = ['blue', 'orange', 'green', 'brown']
    for idx, model in enumerate(['cnn_slider', 'pulse_template_matching', 'resnet34_CAM', 'resnet34_CAM']):
        dices = DICE_dict[model]
        yerrs = []
        ys = []
        s = x_loc
        for dice in dices:
            yerrs.append(float(dice.split(' +- ')[1]))
            ys.append(float(dice.split(' +- ')[0]))
            x_loc += 0.3
            ax.errorbar(x_loc, float(dice.split(' +- ')[0]), yerr=float(dice.split(' +- ')[1]), capsize=3, color=colors[idx])
        tick_pos.append(((x_loc - s) // 2) + s)
        # print(model, len(dices))
        x_loc += 2

    # rgb_colors = []
    # for c in colors:
    #     rgb_colors.append(matplotlib.colors.to_rgba_array(c))

    ax.errorbar(x_loc//2, y=float(DICE_dict['proposed'][0].split(' +- ')[0]), yerr=float(DICE_dict['proposed'][0].split(' +- ')[1]), color='purple', linewidth=4, label='Proposed\nModel')
    ax.axhline(y=float(DICE_dict['proposed'][0].split(' +- ')[0]), color='purple', linewidth=1)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(['Baseline 1', 'Baseline 2', 'Baseline 3', 'Baseline 4'], rotation=45)
    ax.set_ylabel('DICE')

    for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)
    ax.set_title(real_dataset_name)

    # plt.show()


if __name__ == '__main__':
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(10, 4))
    axs = [ax1, ax2, ax3, ax4]
    for idx, dataset_ in enumerate(['ucsf', 'new_PPG_DaLiA_test', 'WESAD_all', 'TROIKA_channel_1']):
        print(dataset_)
        dataset = ''
        if dataset_ == 'new_PPG_DaLiA_test': dataset = 'PPG DaLiA test set'
        if dataset_ == 'WESAD_all': dataset = 'WESAD'
        if dataset_ == 'TROIKA_channel_1': dataset = 'TROIKA'
        if dataset_ == 'ucsf': dataset = 'UCSF'
        get_dice(dataset_, dataset, axs[idx])
        print('----------------')

    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper left', facecolor='white', framealpha=1)

    for text in lgd.get_texts():
        text.set_color("purple")
    plt.tight_layout()
    plt.savefig('visualize_all/all_datasets_DICE-tv2.jpg')




