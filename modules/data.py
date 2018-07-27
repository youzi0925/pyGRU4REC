# coding=utf-8
import numpy as np
import pandas as pd
import torch


class SessionDataset:
    def __init__(self, path, sep='\t', session_key='SessionId', item_key='ItemId', time_key='TimeStamp', n_samples=-1,
                 itemmap=None, time_sort=False):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.df = pd.read_csv(path, sep=sep, names=[session_key, item_key, time_key])
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort

        # sampling
        if n_samples > 0: self.df = self.df[:n_samples]
        # Add item indices
        self.add_item_indices(itemmap=itemmap)
        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        self.df.sort_values([session_key, time_key], inplace=True)

        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = self.order_session_idx()

    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        print(offsets)

        return offsets

    def order_session_idx(self):
        """ Order the session indices """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())

        print(session_idx_arr)
        return session_idx_arr

    def add_item_indices(self, itemmap=None):
        """ 
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                    'item_idx': item2idx[item_ids].values})

        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    @property
    def items(self):
        return self.itemmap.ItemId.unique()


class SessionDataLoader:
    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.

        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        # session_idx_arr[iters]指一个会话id(已经排序)，[click_offsets[session_idx_arr[iters]], click_offsets[session_idx_arr[iters] + 1])
        # 指the offset values,一个session内的数据
        start = click_offsets[session_idx_arr[iters]]  # 一个batch-size的start-click_offsets
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False
        print(df)

        while not finished:
            # 重要,理解,此值为一个batch中所有session的最小长度,注意,start,end为click_offsets数组
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            # 注意item_idx是在groupby session和排序前标上的
            idx_target = df.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                # yield,所有此print为最终循环输出
                print(idx_input)  # len为batch-size  [31 26 27 29 24]
                print(idx_target)  # len为batch-size  [31 26 28 17 24]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            # 理解,当有一个idx_input索引走到了某个session的倒数第二个位置--因为minlen,所以上面循环退出后,有一个session的idx_input一定走到了倒数第二个位置
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            # 一个batch中,哪几个session结束了
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1  # 计数已经训练过的会话总数
                if maxiter >= len(click_offsets) - 1:  # len(click_offsets) - 1:数据中总的会话数
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
