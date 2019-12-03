import os
import csv
import skvideo.io
import numpy as np
import pandas as pd

import lib

import matplotlib.pyplot as plt


__all__ = ['TrainingPlot', 'VideoMaker']


class TrainingPlot(object):
    def __init__(self, file_name, save, fig_size):
        self.file_name = file_name
        self.save = save
        self.fig_size = fig_size

    def plot(self):
        raise NotImplementedError()

    def _plot(self, x, his, key, legend, save_name, title, axis_name, save = True):
        plt.figure(figsize = self.fig_size)
        for i in range(len(his)):
            plt.plot(x, np.array(list(his[i][key])))

        plt.title(title)
        plt.xlabel(axis_name[0])
        plt.ylabel(axis_name[1])
        plt.legend(legend, loc = 'upper right')
        if save:
            plt.savefig(save_name + '.png')
            print('Picture: ' + save_name + '.png done.')
        else:
            plt.show()

        return None

    def _read_history(self, file_list):
        history = []
        longest = 0
        for i in range(len(file_list)):
            df = pd.read_csv(file_list[i])
            if len(df) > longest:
                longest = len(df)

            history.append(df)

        return history, longest

    def _get_filenames(self, name_list, file_name):
        files = []
        if file_name is None:
            for name in name_list:
                files.append(os.path.join(name, name + '.csv'))
        else:
            for name in name_list:
                files.append(os.path.join(name, self.file_name))

        return files


class VideoMaker(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.save_path = self._create_dir(model_name)
        self.video = None
        self.frames = []

    def insert_video(self, new):
        if type(new) != np.ndarray:
            raise TypeError('Please insert numpy array video.')

        if len(new.shape) != 4:
            raise IndexError('Please insert correct shape video, (frames, width, hight, channel)')

        if self.video is None:
            self.video = new
        else:
            self.video = np.concatenate((self.video, new), axis = 0)

        return None

    def insert_frame(self, new):
        if type(new) != np.ndarray:
            raise TypeError('Please insert numpy array frame.')

        self.frames.append(new)

        return None

    def make(self, path, name, delete = True):
        save_path = os.path.join(path, 'video', name + '.mp4')
        self.video = self._build_video(self.video, self.frames)
        skvideo.io.vwrite(save_path, self.video)
        print('Video:', save_path, 'writing done.')

        if delete:
            self.video = None

        return None

    def _build_video(self, video, frames):
        if video is None:
            return np.asarray(frames)
        elif video is not None and len(frames) == 0:
            return video
        else:
            video = np.concatenate((video, np.asarray(frames)), axis = 0)
            return video

    def _create_dir(self, model_name):
        video_path = os.path.join(model_name, 'video')
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        print('All output video will save in', video_path)
        return video_path


