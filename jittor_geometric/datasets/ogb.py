import pandas as pd
import shutil, os
import os.path as osp
import numpy as np
import jittor as jt
from jittor_geometric.io import read_graph, read_heterograph, read_node_label_hetero, read_nodesplitidx_split_hetero
from jittor_geometric.data import InMemoryDataset, download_url, decide_download, extract_zip


class OGBNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                         but when something is passed, it uses its information. Useful for debugging for external contributors.
        ''' 

        self.name = name  # original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name)):
                self.dir_name = self.dir_name

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  # name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(OGBNodePropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = jt.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pkl exists
        if os.path.isfile(os.path.join(path, 'split_dict.pkl')):
            return jt.load(os.path.join(path, 'split_dict.pkl'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = jt.array(train_idx_dict[nodetype]).int32()
                valid_idx_dict[nodetype] = jt.array(valid_idx_dict[nodetype]).int32()
                test_idx_dict[nodetype] = jt.array(test_idx_dict[nodetype]).int32()

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = jt.array(pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]).int32()
            valid_idx = jt.array(pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]).int32()
            test_idx = jt.array(pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]).int32()

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pkl')

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:
            data = read_heterograph(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files, additional_edge_files=additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {key: tmp[key] for key in tmp.keys()}
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = jt.array(node_label).float32()
                    else:
                        data.y_dict[nodetype] = jt.array(node_label).int32()
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = jt.array(node_label).float32()

        else:
            data = read_graph(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files, additional_edge_files=additional_edge_files, binary=self.binary)[0]

            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip', header=None).values

            if 'classification' in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = jt.array(node_label).float32()
                else:
                    data.y = jt.array(node_label).int32()

            else:
                data.y = jt.array(node_label).float32()

        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        jt.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    dataset = OGBNodePropPredDataset(name='ogbn-mag')
    split_index = dataset.get_idx_split()
