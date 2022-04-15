import argparse
import json
import os
import os.path
from collections import defaultdict
from utils.file_managment import FileOp
import numpy as np
from numpy import trapz
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Plot():

    def __init__(self, gt_path, predictions_path, output_path):
        self._gt_path = gt_path
        self._predictions_path = predictions_path
        self._output_path = output_path

    def read_points(self,dir_path, max_points, point_f):
        print('Reading directory {}'.format(dir_path))
        points = {}
        files = os.listdir(dir_path)
        pts = list(filter(lambda x: x.endswith('.pts'), files))

        for  fname in pts:
            if fname in point_f:

                cur_path = os.path.join(dir_path, fname)
                print(cur_path)
                if not os.path.exists(dir_path):
                    print('Wrong directory')
                    break

                if cur_path.endswith('.pts') or cur_path.endswith('.pts1'):

                    with open(cur_path) as cur_file:
                        lines = cur_file.readlines()
                        if lines[0].startswith('version'):  # to support different formats
                            lines = lines[3:-1]
                        mat = np.fromstring(''.join(lines), sep=' ')
                        points[fname] = (mat[0::2], mat[1::2])

            else:
                print('Skip {} Not in Names'.format(fname))

        return points

    def count_ced(self,predicted_points, gt_points):
        ceds = defaultdict(list)

        for method_name in predicted_points.keys():
            print('Counting ces. Method name {}'.format(method_name))
            for pr_points in predicted_points[method_name].keys():
                if pr_points in gt_points:
                    # print('Processing key {}'.format(img_name))
                    x_pred, y_pred = predicted_points[method_name][pr_points]
                    x_gt, y_gt = gt_points[pr_points]
                    n_points = x_pred.shape[0]
                    #assert n_points == x_gt.shape[0], '{} != {}'.format(n_points, x_gt.shape[0])
                    if n_points == x_gt.shape[0]:
                        w = np.max(x_gt) - np.min(x_gt)
                        h = np.max(y_gt) - np.min(y_gt)
                        normalization_factor = np.sqrt(h * w)
                        diff_x = [x_gt[i] - x_pred[i] for i in range(n_points)]
                        diff_y = [y_gt[i] - y_pred[i] for i in range(n_points)]
                        dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
                        avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)
                        ceds[method_name].append(avg_norm_dist)
                    else:
                       print(f'Not equal to {n_points}')


                    # print('Average distance for method {} = {}'.format(method_name, avg_norm_dist))
                else:
                    print('Skipping key {}, because its not in the gt points'.format(pr_points))
            ceds[method_name] = np.sort(ceds[method_name])

        return ceds

    def count_ced_auc(self,errors):
        if not isinstance(errors, list):
            errors = [errors]

        aucs = []
        for error in errors:
            auc = 0
            proportions = np.arange(error.shape[0], dtype=np.float32) / error.shape[0]
            assert (len(proportions) > 0)

            step = 0.01
            for thr in np.arange(0.0, 1.0, step):
                gt_indexes = [idx for idx, e in enumerate(error) if e >= thr]
                if len(gt_indexes) > 0:
                    first_gt_idx = gt_indexes[0]
                else:
                    first_gt_idx = len(error) - 1
                auc += proportions[first_gt_idx] * step
            aucs.append(auc)
        return aucs

    def main(self):


        max_points_to_read = 68
        error_thr = 0.08

        print('error threshold = {}'.format(error_thr))


        files = FileOp(self._predictions_path)
        im, pts, fullpath = files.get_files_from_dir()

        predicted_points = {}

        predicted_points[os.path.basename(self._predictions_path)] = \
            self.read_points(self._predictions_path, max_points_to_read, pts)

        gt_points = self.read_points(self._gt_path, max_points_to_read, pts)


        ceds = self.count_ced(predicted_points, gt_points)

        # saving figure
        line_styles = [':', '-.', '--', '-']
        plt.figure(figsize=(30, 20), dpi=100)
        for method_idx, method_name in enumerate(ceds.keys()):
            print('Plotting graph for the method {}'.format(method_name))
            err = ceds[method_name]
            proportion = np.arange(err.shape[0], dtype=np.float32) / err.shape[0]
            under_thr = err > error_thr
            last_idx = len(err)
            if len(np.flatnonzero(under_thr)) > 0:
                last_idx = np.flatnonzero(under_thr)[0]
            under_thr_range = range(last_idx)
            cur_auc = self.count_ced_auc(err)[0]

            area = trapz(err[under_thr_range], proportion[under_thr_range])
            print(area)
            plt.plot(err[under_thr_range], proportion[under_thr_range], label=method_name +
                                                                              ', auc={:1.3f},area={}'.format(cur_auc,
                                                                                                             area),
                     linestyle=line_styles[method_idx % len(line_styles)], linewidth=2.0)
        plt.legend(loc='right', prop={'size': 24})
        plt.savefig(self._output_path)


